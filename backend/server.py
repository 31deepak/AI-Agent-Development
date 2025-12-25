from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import re
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class LineItem(BaseModel):
    sku: Optional[str] = None
    description: Optional[str] = None
    qty: float
    unitPrice: float
    qtyDelivered: Optional[float] = None

class InvoiceFields(BaseModel):
    invoiceNumber: str
    invoiceDate: str
    serviceDate: Optional[str] = None
    currency: Optional[str] = None
    poNumber: Optional[str] = None
    netTotal: float
    taxRate: float
    taxTotal: float
    grossTotal: float
    lineItems: List[LineItem]
    discountTerms: Optional[str] = None

class Invoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    invoiceId: str
    vendor: str
    fields: InvoiceFields
    confidence: float
    rawText: str

class AuditEntry(BaseModel):
    step: str  # recall|apply|decide|learn
    timestamp: str
    details: str

class MemoryUpdate(BaseModel):
    memoryType: str
    action: str
    key: str
    value: Any
    confidence: float

class ProposedCorrection(BaseModel):
    field: str
    fromValue: Any
    toValue: Any
    reason: str
    confidence: float

class ProcessingResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    invoiceId: str
    vendor: str
    normalizedInvoice: Dict[str, Any]
    proposedCorrections: List[Dict[str, Any]]
    requiresHumanReview: bool
    reasoning: str
    confidenceScore: float
    memoryUpdates: List[Dict[str, Any]]
    auditTrail: List[Dict[str, Any]]
    decision: str  # auto-accepted|auto-corrected|escalated
    status: str  # pending|approved|rejected
    processedAt: str
    originalInvoice: Dict[str, Any]

class HumanCorrectionInput(BaseModel):
    invoiceId: str
    corrections: List[Dict[str, Any]]
    finalDecision: str  # approved|rejected

class VendorMemory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor: str
    patterns: Dict[str, Any]  # e.g., {"serviceDateLabel": "Leistungsdatum", "vatBehavior": "included"}
    confidence: float
    usageCount: int
    successCount: int
    lastUsed: str
    createdAt: str

class CorrectionMemory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor: str
    field: str
    pattern: str  # e.g., "qty_mismatch_adjust_to_dn"
    correctionRule: Dict[str, Any]
    confidence: float
    usageCount: int
    successCount: int
    lastUsed: str
    createdAt: str

class ResolutionMemory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor: str
    discrepancyType: str
    resolution: str  # approved|rejected
    count: int
    lastOccurrence: str
    createdAt: str

# ==================== MEMORY ENGINE ====================

class MemoryEngine:
    def __init__(self, db):
        self.db = db
        self.CONFIDENCE_THRESHOLD_AUTO = 0.80
        self.CONFIDENCE_THRESHOLD_ESCALATE = 0.40
        self.DECAY_RATE = 0.02  # per week
        self.REINFORCEMENT_RATE = 0.10
        self.MAX_CONFIDENCE = 0.95
        self.MIN_CONFIDENCE = 0.10

    async def recall_memory(self, invoice: Dict, audit_trail: List) -> Dict:
        """Retrieve relevant past learnings for the invoice context"""
        vendor = invoice.get('vendor', '')
        
        # Get vendor memory
        vendor_memory = await self.db.vendor_memory.find_one(
            {"vendor": vendor}, {"_id": 0}
        )
        
        # Get correction memories for this vendor
        correction_memories = await self.db.correction_memory.find(
            {"vendor": vendor}, {"_id": 0}
        ).to_list(100)
        
        # Get resolution memories
        resolution_memories = await self.db.resolution_memory.find(
            {"vendor": vendor}, {"_id": 0}
        ).to_list(100)
        
        # Check for duplicate invoices
        duplicate_check = await self._check_duplicates(invoice)
        
        audit_trail.append({
            "step": "recall",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": f"Retrieved {1 if vendor_memory else 0} vendor memory, {len(correction_memories)} correction memories, {len(resolution_memories)} resolution memories for {vendor}. Duplicate check: {duplicate_check['isDuplicate']}"
        })
        
        return {
            "vendorMemory": vendor_memory,
            "correctionMemories": correction_memories,
            "resolutionMemories": resolution_memories,
            "duplicateCheck": duplicate_check
        }

    async def _check_duplicates(self, invoice: Dict) -> Dict:
        """Check if this invoice is a duplicate"""
        vendor = invoice.get('vendor', '')
        invoice_number = invoice.get('fields', {}).get('invoiceNumber', '')
        invoice_date = invoice.get('fields', {}).get('invoiceDate', '')
        
        # Find existing processed invoices with same vendor and invoice number
        existing = await self.db.processed_invoices.find(
            {
                "vendor": vendor,
                "normalizedInvoice.invoiceNumber": invoice_number,
                "invoiceId": {"$ne": invoice.get('invoiceId')}
            },
            {"_id": 0}
        ).to_list(10)
        
        if existing:
            return {
                "isDuplicate": True,
                "existingInvoices": [e.get('invoiceId') for e in existing],
                "reason": f"Invoice {invoice_number} from {vendor} already exists"
            }
        
        return {"isDuplicate": False, "existingInvoices": [], "reason": None}

    async def apply_memory(self, invoice: Dict, recalled: Dict, audit_trail: List) -> Dict:
        """Apply memory to normalize fields and suggest corrections"""
        normalized = invoice.get('fields', {}).copy()
        corrections = []
        memory_updates = []
        reasoning_parts = []
        
        vendor_memory = recalled.get('vendorMemory')
        correction_memories = recalled.get('correctionMemories', [])
        raw_text = invoice.get('rawText', '')
        vendor = invoice.get('vendor', '')
        
        # Apply vendor-specific patterns
        if vendor_memory:
            patterns = vendor_memory.get('patterns', {})
            vm_confidence = vendor_memory.get('confidence', 0.5)
            
            # Service date extraction from raw text
            if patterns.get('serviceDateLabel') and not normalized.get('serviceDate'):
                label = patterns['serviceDateLabel']
                date_match = re.search(rf'{label}[:\s]+([\d./-]+)', raw_text)
                if date_match:
                    extracted_date = self._parse_date(date_match.group(1))
                    if extracted_date and vm_confidence >= 0.5:
                        corrections.append({
                            "field": "serviceDate",
                            "fromValue": None,
                            "toValue": extracted_date,
                            "reason": f"Extracted from '{label}' in raw text (vendor pattern)",
                            "confidence": vm_confidence
                        })
                        normalized['serviceDate'] = extracted_date
                        reasoning_parts.append(f"Applied vendor memory: '{label}' -> serviceDate")
            
            # VAT behavior handling
            if patterns.get('vatBehavior') == 'included':
                # Check if rawText indicates VAT is included
                vat_included_patterns = ['vat included', 'vat already included', 'mwst. inkl', 'prices incl. vat', 'inkl. mwst']
                if any(p in raw_text.lower() for p in vat_included_patterns):
                    # Recalculate tax from gross total
                    gross = normalized.get('grossTotal', 0)
                    tax_rate = normalized.get('taxRate', 0.19)
                    calculated_net = gross / (1 + tax_rate)
                    calculated_tax = gross - calculated_net
                    
                    if abs(calculated_tax - normalized.get('taxTotal', 0)) > 0.01:
                        corrections.append({
                            "field": "taxTotal",
                            "fromValue": normalized.get('taxTotal'),
                            "toValue": round(calculated_tax, 2),
                            "reason": "VAT included in total; recalculated based on vendor pattern",
                            "confidence": vm_confidence
                        })
                        corrections.append({
                            "field": "netTotal",
                            "fromValue": normalized.get('netTotal'),
                            "toValue": round(calculated_net, 2),
                            "reason": "Recalculated from gross (VAT included)",
                            "confidence": vm_confidence
                        })
                        normalized['taxTotal'] = round(calculated_tax, 2)
                        normalized['netTotal'] = round(calculated_net, 2)
                        reasoning_parts.append("Applied VAT-included pattern from vendor memory")
        
        # Extract currency from raw text if missing
        if not normalized.get('currency'):
            currency_match = re.search(r'Currency[:\s]+(EUR|USD|GBP)', raw_text, re.IGNORECASE)
            if currency_match:
                confidence = 0.7 if vendor_memory else 0.5
                corrections.append({
                    "field": "currency",
                    "fromValue": None,
                    "toValue": currency_match.group(1).upper(),
                    "reason": "Currency found in raw text",
                    "confidence": confidence
                })
                normalized['currency'] = currency_match.group(1).upper()
                reasoning_parts.append("Extracted currency from raw text")
        
        # Extract discount terms (Skonto) from raw text
        skonto_match = re.search(r'(\d+)%\s*[Ss]konto.*?(\d+)\s*[Dd]ays?', raw_text)
        if skonto_match and not normalized.get('discountTerms'):
            terms = f"{skonto_match.group(1)}% Skonto within {skonto_match.group(2)} days"
            corrections.append({
                "field": "discountTerms",
                "fromValue": None,
                "toValue": terms,
                "reason": "Skonto terms detected in raw text",
                "confidence": 0.8
            })
            normalized['discountTerms'] = terms
            reasoning_parts.append("Extracted discount terms from raw text")
        
        # Try to match PO if missing
        if not normalized.get('poNumber'):
            po_suggestion = await self._suggest_po_match(invoice, vendor)
            if po_suggestion:
                corrections.append({
                    "field": "poNumber",
                    "fromValue": None,
                    "toValue": po_suggestion['poNumber'],
                    "reason": po_suggestion['reason'],
                    "confidence": po_suggestion['confidence']
                })
                normalized['poNumber'] = po_suggestion['poNumber']
                reasoning_parts.append(f"Suggested PO match: {po_suggestion['poNumber']}")
        
        # Map description to SKU based on correction memory
        for cm in correction_memories:
            if cm.get('pattern') == 'description_to_sku' and cm.get('confidence', 0) >= 0.6:
                rule = cm.get('correctionRule', {})
                for i, item in enumerate(normalized.get('lineItems', [])):
                    desc = item.get('description', '').lower()
                    if not item.get('sku') and rule.get('descriptionPattern', '').lower() in desc:
                        corrections.append({
                            "field": f"lineItems[{i}].sku",
                            "fromValue": None,
                            "toValue": rule.get('sku'),
                            "reason": f"Description '{item.get('description')}' maps to SKU based on learned pattern",
                            "confidence": cm.get('confidence', 0.6)
                        })
                        normalized['lineItems'][i]['sku'] = rule.get('sku')
                        reasoning_parts.append(f"Applied SKU mapping from correction memory")
        
        audit_trail.append({
            "step": "apply",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": f"Applied {len(corrections)} corrections. Reasoning: {'; '.join(reasoning_parts) if reasoning_parts else 'No applicable memories'}"
        })
        
        return {
            "normalized": normalized,
            "corrections": corrections,
            "reasoning": reasoning_parts,
            "memoryUpdates": memory_updates
        }

    async def _suggest_po_match(self, invoice: Dict, vendor: str) -> Optional[Dict]:
        """Try to find a matching PO for the invoice"""
        # Get POs for this vendor
        pos = await self.db.purchase_orders.find(
            {"vendor": vendor}, {"_id": 0}
        ).to_list(100)
        
        if not pos:
            return None
        
        invoice_items = invoice.get('fields', {}).get('lineItems', [])
        invoice_skus = {item.get('sku') for item in invoice_items if item.get('sku')}
        
        for po in pos:
            po_items = po.get('lineItems', [])
            po_skus = {item.get('sku') for item in po_items if item.get('sku')}
            
            # Check SKU overlap
            if invoice_skus and po_skus and invoice_skus.intersection(po_skus):
                return {
                    "poNumber": po.get('poNumber'),
                    "reason": f"Matching SKU found in PO {po.get('poNumber')} within vendor {vendor}",
                    "confidence": 0.75
                }
        
        # If only one PO exists for vendor, suggest it with lower confidence
        if len(pos) == 1:
            return {
                "poNumber": pos[0].get('poNumber'),
                "reason": f"Only matching PO for vendor {vendor}",
                "confidence": 0.5
            }
        
        return None

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats to ISO format"""
        formats = [
            '%d.%m.%Y', '%d-%m-%Y', '%Y-%m-%d',
            '%d/%m/%Y', '%m/%d/%Y'
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    async def decide(self, invoice: Dict, recalled: Dict, applied: Dict, audit_trail: List) -> Dict:
        """Make decision: auto-accept, auto-correct, or escalate"""
        corrections = applied.get('corrections', [])
        duplicate_check = recalled.get('duplicateCheck', {})
        
        # Check for duplicates first
        if duplicate_check.get('isDuplicate'):
            audit_trail.append({
                "step": "decide",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": f"ESCALATE: Duplicate invoice detected. {duplicate_check.get('reason')}"
            })
            return {
                "decision": "escalated",
                "requiresHumanReview": True,
                "confidenceScore": 0.0,
                "reasoning": f"Duplicate detected: {duplicate_check.get('reason')}"
            }
        
        # Calculate overall confidence
        base_confidence = invoice.get('confidence', 0.5)
        
        if not corrections:
            # No corrections needed
            if base_confidence >= self.CONFIDENCE_THRESHOLD_AUTO:
                decision = "auto-accepted"
                requires_review = False
                reasoning = "High extraction confidence, no corrections needed"
            elif base_confidence >= self.CONFIDENCE_THRESHOLD_ESCALATE:
                decision = "auto-accepted"
                requires_review = False
                reasoning = "Acceptable extraction confidence, no corrections needed"
            else:
                decision = "escalated"
                requires_review = True
                reasoning = "Low extraction confidence, requires human review"
        else:
            # Calculate average correction confidence
            avg_correction_confidence = sum(c.get('confidence', 0.5) for c in corrections) / len(corrections)
            combined_confidence = (base_confidence + avg_correction_confidence) / 2
            
            # Check if any correction has low confidence
            low_confidence_corrections = [c for c in corrections if c.get('confidence', 0) < 0.6]
            
            if combined_confidence >= self.CONFIDENCE_THRESHOLD_AUTO and not low_confidence_corrections:
                decision = "auto-corrected"
                requires_review = False
                reasoning = f"Applied {len(corrections)} high-confidence corrections automatically"
            elif combined_confidence >= self.CONFIDENCE_THRESHOLD_ESCALATE:
                decision = "auto-corrected"
                requires_review = True
                reasoning = f"Applied {len(corrections)} corrections but some have lower confidence - review recommended"
            else:
                decision = "escalated"
                requires_review = True
                reasoning = f"Low confidence corrections ({len(low_confidence_corrections)} below threshold) - human review required"
            
            base_confidence = combined_confidence
        
        audit_trail.append({
            "step": "decide",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": f"{decision.upper()}: {reasoning}. Confidence: {base_confidence:.2f}"
        })
        
        return {
            "decision": decision,
            "requiresHumanReview": requires_review,
            "confidenceScore": round(base_confidence, 3),
            "reasoning": reasoning
        }

    async def learn(self, invoice_id: str, corrections: List[Dict], final_decision: str, audit_trail: List) -> List[Dict]:
        """Store new insights and reinforce/weaken existing memories"""
        memory_updates = []
        
        # Get the processed invoice
        processed = await self.db.processed_invoices.find_one(
            {"invoiceId": invoice_id}, {"_id": 0}
        )
        
        if not processed:
            return memory_updates
        
        vendor = processed.get('vendor', '')
        now = datetime.now(timezone.utc).isoformat()
        
        for correction in corrections:
            field = correction.get('field', '')
            to_value = correction.get('to')
            reason = correction.get('reason', '')
            
            # Learn service date pattern
            if field == 'serviceDate' and 'leistungsdatum' in reason.lower():
                await self._update_vendor_memory(
                    vendor,
                    'serviceDateLabel',
                    'Leistungsdatum',
                    final_decision == 'approved'
                )
                memory_updates.append({
                    "memoryType": "vendor",
                    "action": "update" if final_decision == 'approved' else "weaken",
                    "key": f"{vendor}/serviceDateLabel",
                    "value": "Leistungsdatum",
                    "confidence": 0.0  # Will be set by _update_vendor_memory
                })
            
            # Learn VAT behavior
            if field in ['taxTotal', 'grossTotal', 'netTotal'] and 'vat' in reason.lower():
                await self._update_vendor_memory(
                    vendor,
                    'vatBehavior',
                    'included',
                    final_decision == 'approved'
                )
                memory_updates.append({
                    "memoryType": "vendor",
                    "action": "update" if final_decision == 'approved' else "weaken",
                    "key": f"{vendor}/vatBehavior",
                    "value": "included",
                    "confidence": 0.0
                })
            
            # Learn SKU mapping
            if '.sku' in field and to_value:
                # Extract description from the correction reason
                desc_match = re.search(r"Description '([^']+)'", reason)
                if desc_match or 'seefracht' in reason.lower() or 'shipping' in reason.lower():
                    description = desc_match.group(1) if desc_match else 'Transport/Shipping'
                    await self._update_correction_memory(
                        vendor,
                        'description_to_sku',
                        {
                            "descriptionPattern": description.split('/')[0].strip().lower(),
                            "sku": to_value
                        },
                        final_decision == 'approved'
                    )
                    memory_updates.append({
                        "memoryType": "correction",
                        "action": "update" if final_decision == 'approved' else "weaken",
                        "key": f"{vendor}/description_to_sku",
                        "value": {"description": description, "sku": to_value},
                        "confidence": 0.0
                    })
            
            # Learn PO matching pattern
            if field == 'poNumber' and to_value:
                await self._update_correction_memory(
                    vendor,
                    'po_matching',
                    {"poNumber": to_value},
                    final_decision == 'approved'
                )
                memory_updates.append({
                    "memoryType": "correction",
                    "action": "update" if final_decision == 'approved' else "weaken",
                    "key": f"{vendor}/po_matching",
                    "value": to_value,
                    "confidence": 0.0
                })
        
        # Update resolution memory
        await self._update_resolution_memory(vendor, 'general', final_decision)
        memory_updates.append({
            "memoryType": "resolution",
            "action": "record",
            "key": f"{vendor}/general",
            "value": final_decision,
            "confidence": 0.0
        })
        
        audit_trail.append({
            "step": "learn",
            "timestamp": now,
            "details": f"Stored {len(memory_updates)} memory updates from {final_decision} decision"
        })
        
        return memory_updates

    async def _update_vendor_memory(self, vendor: str, key: str, value: Any, success: bool):
        """Update or create vendor memory"""
        now = datetime.now(timezone.utc).isoformat()
        existing = await self.db.vendor_memory.find_one({"vendor": vendor})
        
        if existing:
            patterns = existing.get('patterns', {})
            patterns[key] = value
            
            new_confidence = existing.get('confidence', 0.5)
            if success:
                new_confidence = min(self.MAX_CONFIDENCE, new_confidence + self.REINFORCEMENT_RATE)
            else:
                new_confidence = max(self.MIN_CONFIDENCE, new_confidence - self.REINFORCEMENT_RATE * 2)
            
            await self.db.vendor_memory.update_one(
                {"vendor": vendor},
                {
                    "$set": {
                        "patterns": patterns,
                        "confidence": new_confidence,
                        "lastUsed": now
                    },
                    "$inc": {
                        "usageCount": 1,
                        "successCount": 1 if success else 0
                    }
                }
            )
        else:
            new_memory = {
                "id": str(uuid.uuid4()),
                "vendor": vendor,
                "patterns": {key: value},
                "confidence": 0.6 if success else 0.4,
                "usageCount": 1,
                "successCount": 1 if success else 0,
                "lastUsed": now,
                "createdAt": now
            }
            await self.db.vendor_memory.insert_one(new_memory)

    async def _update_correction_memory(self, vendor: str, pattern: str, rule: Dict, success: bool):
        """Update or create correction memory"""
        now = datetime.now(timezone.utc).isoformat()
        existing = await self.db.correction_memory.find_one({
            "vendor": vendor,
            "pattern": pattern
        })
        
        if existing:
            new_confidence = existing.get('confidence', 0.5)
            if success:
                new_confidence = min(self.MAX_CONFIDENCE, new_confidence + self.REINFORCEMENT_RATE)
            else:
                new_confidence = max(self.MIN_CONFIDENCE, new_confidence - self.REINFORCEMENT_RATE * 2)
            
            await self.db.correction_memory.update_one(
                {"vendor": vendor, "pattern": pattern},
                {
                    "$set": {
                        "correctionRule": rule,
                        "confidence": new_confidence,
                        "lastUsed": now
                    },
                    "$inc": {
                        "usageCount": 1,
                        "successCount": 1 if success else 0
                    }
                }
            )
        else:
            new_memory = {
                "id": str(uuid.uuid4()),
                "vendor": vendor,
                "field": rule.get('field', 'general'),
                "pattern": pattern,
                "correctionRule": rule,
                "confidence": 0.6 if success else 0.4,
                "usageCount": 1,
                "successCount": 1 if success else 0,
                "lastUsed": now,
                "createdAt": now
            }
            await self.db.correction_memory.insert_one(new_memory)

    async def _update_resolution_memory(self, vendor: str, discrepancy_type: str, resolution: str):
        """Update or create resolution memory"""
        now = datetime.now(timezone.utc).isoformat()
        existing = await self.db.resolution_memory.find_one({
            "vendor": vendor,
            "discrepancyType": discrepancy_type,
            "resolution": resolution
        })
        
        if existing:
            await self.db.resolution_memory.update_one(
                {"vendor": vendor, "discrepancyType": discrepancy_type, "resolution": resolution},
                {
                    "$set": {"lastOccurrence": now},
                    "$inc": {"count": 1}
                }
            )
        else:
            new_memory = {
                "id": str(uuid.uuid4()),
                "vendor": vendor,
                "discrepancyType": discrepancy_type,
                "resolution": resolution,
                "count": 1,
                "lastOccurrence": now,
                "createdAt": now
            }
            await self.db.resolution_memory.insert_one(new_memory)

    async def apply_decay(self):
        """Apply confidence decay to old memories"""
        cutoff = (datetime.now(timezone.utc) - timedelta(weeks=1)).isoformat()
        
        # Decay vendor memories
        await self.db.vendor_memory.update_many(
            {"lastUsed": {"$lt": cutoff}},
            {"$mul": {"confidence": (1 - self.DECAY_RATE)}}
        )
        
        # Decay correction memories
        await self.db.correction_memory.update_many(
            {"lastUsed": {"$lt": cutoff}},
            {"$mul": {"confidence": (1 - self.DECAY_RATE)}}
        )

memory_engine = MemoryEngine(db)

# ==================== API ENDPOINTS ====================

@api_router.get("/")
async def root():
    return {"message": "AI Memory Layer for Invoice Processing"}

@api_router.post("/invoices/process")
async def process_invoice(invoice: Invoice):
    """Process a single invoice through the memory layer"""
    audit_trail = []
    invoice_dict = invoice.model_dump()
    
    # Step 1: Recall
    recalled = await memory_engine.recall_memory(invoice_dict, audit_trail)
    
    # Step 2: Apply
    applied = await memory_engine.apply_memory(invoice_dict, recalled, audit_trail)
    
    # Step 3: Decide
    decision = await memory_engine.decide(invoice_dict, recalled, applied, audit_trail)
    
    # Create processing result
    result = {
        "id": str(uuid.uuid4()),
        "invoiceId": invoice.invoiceId,
        "vendor": invoice.vendor,
        "normalizedInvoice": applied['normalized'],
        "proposedCorrections": applied['corrections'],
        "requiresHumanReview": decision['requiresHumanReview'],
        "reasoning": decision['reasoning'] + ". " + "; ".join(applied['reasoning']) if applied['reasoning'] else decision['reasoning'],
        "confidenceScore": decision['confidenceScore'],
        "memoryUpdates": [],
        "auditTrail": audit_trail,
        "decision": decision['decision'],
        "status": "pending" if decision['requiresHumanReview'] else "auto-processed",
        "processedAt": datetime.now(timezone.utc).isoformat(),
        "originalInvoice": invoice_dict
    }
    
    # Store in database
    await db.processed_invoices.insert_one(result.copy())
    
    return result

@api_router.post("/invoices/batch-process")
async def batch_process_invoices(invoices: List[Invoice]):
    """Process multiple invoices"""
    results = []
    for invoice in invoices:
        result = await process_invoice(invoice)
        results.append(result)
    return {"processed": len(results), "results": results}

@api_router.get("/invoices")
async def get_invoices():
    """Get all processed invoices"""
    invoices = await db.processed_invoices.find({}, {"_id": 0}).to_list(1000)
    return invoices

@api_router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    """Get a specific processed invoice"""
    invoice = await db.processed_invoices.find_one(
        {"invoiceId": invoice_id}, {"_id": 0}
    )
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice

@api_router.post("/corrections")
async def submit_correction(correction: HumanCorrectionInput):
    """Submit human correction and trigger learning"""
    invoice_id = correction.invoiceId
    
    # Get existing processed invoice
    processed = await db.processed_invoices.find_one(
        {"invoiceId": invoice_id}, {"_id": 0}
    )
    
    if not processed:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    audit_trail = processed.get('auditTrail', [])
    
    # Learn from corrections
    memory_updates = await memory_engine.learn(
        invoice_id,
        correction.corrections,
        correction.finalDecision,
        audit_trail
    )
    
    # Apply corrections to normalized invoice
    normalized = processed.get('normalizedInvoice', {})
    for corr in correction.corrections:
        field = corr.get('field', '')
        to_value = corr.get('to')
        
        if '[' in field:  # Handle array notation like lineItems[0].sku
            match = re.match(r'(\w+)\[(\d+)\]\.(\w+)', field)
            if match:
                arr_name, idx, prop = match.groups()
                if arr_name in normalized and int(idx) < len(normalized[arr_name]):
                    normalized[arr_name][int(idx)][prop] = to_value
        else:
            normalized[field] = to_value
    
    # Update in database
    await db.processed_invoices.update_one(
        {"invoiceId": invoice_id},
        {
            "$set": {
                "normalizedInvoice": normalized,
                "status": correction.finalDecision,
                "auditTrail": audit_trail,
                "memoryUpdates": memory_updates,
                "humanCorrections": correction.corrections
            }
        }
    )
    
    return {
        "success": True,
        "invoiceId": invoice_id,
        "memoryUpdates": memory_updates,
        "finalStatus": correction.finalDecision
    }

@api_router.get("/memory/vendor")
async def get_vendor_memories():
    """Get all vendor memories"""
    memories = await db.vendor_memory.find({}, {"_id": 0}).to_list(100)
    return memories

@api_router.get("/memory/corrections")
async def get_correction_memories():
    """Get all correction memories"""
    memories = await db.correction_memory.find({}, {"_id": 0}).to_list(100)
    return memories

@api_router.get("/memory/resolutions")
async def get_resolution_memories():
    """Get all resolution memories"""
    memories = await db.resolution_memory.find({}, {"_id": 0}).to_list(100)
    return memories

@api_router.get("/memory/stats")
async def get_memory_stats():
    """Get memory statistics"""
    vendor_count = await db.vendor_memory.count_documents({})
    correction_count = await db.correction_memory.count_documents({})
    resolution_count = await db.resolution_memory.count_documents({})
    
    # Get average confidences
    vendor_memories = await db.vendor_memory.find({}, {"_id": 0, "confidence": 1}).to_list(100)
    correction_memories = await db.correction_memory.find({}, {"_id": 0, "confidence": 1}).to_list(100)
    
    avg_vendor_conf = sum(m.get('confidence', 0) for m in vendor_memories) / max(len(vendor_memories), 1)
    avg_correction_conf = sum(m.get('confidence', 0) for m in correction_memories) / max(len(correction_memories), 1)
    
    return {
        "vendorMemoryCount": vendor_count,
        "correctionMemoryCount": correction_count,
        "resolutionMemoryCount": resolution_count,
        "avgVendorConfidence": round(avg_vendor_conf, 3),
        "avgCorrectionConfidence": round(avg_correction_conf, 3)
    }

@api_router.delete("/memory/clear")
async def clear_all_memories():
    """Clear all memories (for demo reset)"""
    await db.vendor_memory.delete_many({})
    await db.correction_memory.delete_many({})
    await db.resolution_memory.delete_many({})
    await db.processed_invoices.delete_many({})
    await db.purchase_orders.delete_many({})
    await db.delivery_notes.delete_many({})
    return {"success": True, "message": "All memories cleared"}

@api_router.post("/demo/initialize")
async def initialize_demo():
    """Initialize demo with sample data"""
    # Clear existing data
    await clear_all_memories()
    
    # Load sample purchase orders
    purchase_orders = [
        {"poNumber": "PO-A-050", "vendor": "Supplier GmbH", "date": "2024-01-05", "lineItems": [{"sku": "WIDGET-001", "qty": 100, "unitPrice": 25.0}]},
        {"poNumber": "PO-A-051", "vendor": "Supplier GmbH", "date": "2024-01-18", "lineItems": [{"sku": "WIDGET-002", "qty": 20, "unitPrice": 25.0}]},
        {"poNumber": "PO-B-110", "vendor": "Parts AG", "date": "2024-02-01", "lineItems": [{"sku": "BOLT-99", "qty": 200, "unitPrice": 10.0}]},
        {"poNumber": "PO-B-111", "vendor": "Parts AG", "date": "2024-03-01", "lineItems": [{"sku": "NUT-10", "qty": 500, "unitPrice": 2.0}]},
        {"poNumber": "PO-C-900", "vendor": "Freight & Co", "date": "2024-03-01", "lineItems": [{"sku": "FREIGHT", "qty": 1, "unitPrice": 1000.0}]},
        {"poNumber": "PO-C-901", "vendor": "Freight & Co", "date": "2024-03-15", "lineItems": [{"sku": "FREIGHT", "qty": 1, "unitPrice": 1000.0}]}
    ]
    
    delivery_notes = [
        {"dnNumber": "DN-A-123", "vendor": "Supplier GmbH", "poNumber": "PO-A-050", "date": "2024-01-10", "lineItems": [{"sku": "WIDGET-001", "qtyDelivered": 95}]},
        {"dnNumber": "DN-A-124", "vendor": "Supplier GmbH", "poNumber": "PO-A-051", "date": "2024-01-22", "lineItems": [{"sku": "WIDGET-002", "qtyDelivered": 20}]},
        {"dnNumber": "DN-B-205", "vendor": "Parts AG", "poNumber": "PO-B-110", "date": "2024-02-03", "lineItems": [{"sku": "BOLT-99", "qtyDelivered": 200}]},
        {"dnNumber": "DN-B-206", "vendor": "Parts AG", "poNumber": "PO-B-111", "date": "2024-03-02", "lineItems": [{"sku": "NUT-10", "qtyDelivered": 500}]},
        {"dnNumber": "DN-C-301", "vendor": "Freight & Co", "poNumber": "PO-C-900", "date": "2024-03-02", "lineItems": [{"sku": "FREIGHT", "qtyDelivered": 1}]},
        {"dnNumber": "DN-C-302", "vendor": "Freight & Co", "poNumber": "PO-C-901", "date": "2024-03-18", "lineItems": [{"sku": "FREIGHT", "qtyDelivered": 1}]}
    ]
    
    await db.purchase_orders.insert_many(purchase_orders)
    await db.delivery_notes.insert_many(delivery_notes)
    
    return {
        "success": True,
        "message": "Demo initialized",
        "purchaseOrders": len(purchase_orders),
        "deliveryNotes": len(delivery_notes)
    }

@api_router.get("/demo/sample-invoices")
async def get_sample_invoices():
    """Get sample invoices for demo"""
    return [
        {"invoiceId": "INV-A-001", "vendor": "Supplier GmbH", "fields": {"invoiceNumber": "INV-2024-001", "invoiceDate": "12.01.2024", "serviceDate": None, "currency": "EUR", "poNumber": "PO-A-050", "netTotal": 2500.0, "taxRate": 0.19, "taxTotal": 475.0, "grossTotal": 2975.0, "lineItems": [{"sku": "WIDGET-001", "description": "Widget", "qty": 100, "unitPrice": 25.0}]}, "confidence": 0.78, "rawText": "Rechnungsnr: INV-2024-001\nLeistungsdatum: 01.01.2024\nBestellnr: PO-A-050\n..."},
        {"invoiceId": "INV-A-002", "vendor": "Supplier GmbH", "fields": {"invoiceNumber": "INV-2024-002", "invoiceDate": "18.01.2024", "serviceDate": None, "currency": "EUR", "poNumber": "PO-A-050", "netTotal": 2375.0, "taxRate": 0.19, "taxTotal": 451.25, "grossTotal": 2826.25, "lineItems": [{"sku": "WIDGET-001", "description": "Widget", "qty": 95, "unitPrice": 25.0}]}, "confidence": 0.72, "rawText": "Rechnungsnr: INV-2024-002\nLeistungsdatum: 15.01.2024\nBestellnr: PO-A-050\nHinweis: Teillieferung\n..."},
        {"invoiceId": "INV-A-003", "vendor": "Supplier GmbH", "fields": {"invoiceNumber": "INV-2024-003", "invoiceDate": "25.01.2024", "serviceDate": None, "currency": "EUR", "poNumber": None, "netTotal": 500.0, "taxRate": 0.19, "taxTotal": 95.0, "grossTotal": 595.0, "lineItems": [{"sku": "WIDGET-002", "description": "Widget Pro", "qty": 20, "unitPrice": 25.0}]}, "confidence": 0.69, "rawText": "Rechnungsnr: INV-2024-003\nLeistungsdatum: 20.01.2024\nBestellung: (keine Angabe)\nReferenz: Lieferung Januar\n..."},
        {"invoiceId": "INV-A-004", "vendor": "Supplier GmbH", "fields": {"invoiceNumber": "INV-2024-003", "invoiceDate": "26.01.2024", "serviceDate": None, "currency": "EUR", "poNumber": None, "netTotal": 500.0, "taxRate": 0.19, "taxTotal": 95.0, "grossTotal": 595.0, "lineItems": [{"sku": "WIDGET-002", "description": "Widget Pro", "qty": 20, "unitPrice": 25.0}]}, "confidence": 0.63, "rawText": "Rechnungsnr: INV-2024-003\nLeistungsdatum: 20.01.2024\nHinweis: erneute Zusendung\n..."},
        {"invoiceId": "INV-B-001", "vendor": "Parts AG", "fields": {"invoiceNumber": "PA-7781", "invoiceDate": "05-02-2024", "currency": "EUR", "poNumber": "PO-B-110", "netTotal": 2000.0, "taxRate": 0.19, "taxTotal": 400.0, "grossTotal": 2400.0, "lineItems": [{"sku": "BOLT-99", "description": "Bolts", "qty": 200, "unitPrice": 10.0}]}, "confidence": 0.74, "rawText": "Invoice No: PA-7781\nPO: PO-B-110\nPrices incl. VAT (MwSt. inkl.)\nTotal: 2380.00 EUR\n..."},
        {"invoiceId": "INV-B-002", "vendor": "Parts AG", "fields": {"invoiceNumber": "PA-7799", "invoiceDate": "20-02-2024", "currency": "EUR", "poNumber": "PO-B-110", "netTotal": 1500.0, "taxRate": 0.19, "taxTotal": 285.0, "grossTotal": 1785.0, "lineItems": [{"sku": "BOLT-99", "description": "Bolts", "qty": 150, "unitPrice": 10.0}]}, "confidence": 0.86, "rawText": "Invoice No: PA-7799\nPO: PO-B-110\nMwSt. inkl.\n..."},
        {"invoiceId": "INV-B-003", "vendor": "Parts AG", "fields": {"invoiceNumber": "PA-7810", "invoiceDate": "03-03-2024", "currency": None, "poNumber": "PO-B-111", "netTotal": 1000.0, "taxRate": 0.19, "taxTotal": 190.0, "grossTotal": 1190.0, "lineItems": [{"sku": "NUT-10", "description": "Nuts", "qty": 500, "unitPrice": 2.0}]}, "confidence": 0.66, "rawText": "Invoice No: PA-7810\nPO: PO-B-111\nCurrency: EUR\n..."},
        {"invoiceId": "INV-B-004", "vendor": "Parts AG", "fields": {"invoiceNumber": "PA-7810", "invoiceDate": "04-03-2024", "currency": "EUR", "poNumber": "PO-B-111", "netTotal": 1000.0, "taxRate": 0.19, "taxTotal": 190.0, "grossTotal": 1190.0, "lineItems": [{"sku": "NUT-10", "description": "Nuts", "qty": 500, "unitPrice": 2.0}]}, "confidence": 0.61, "rawText": "Duplicate submission of PA-7810\n..."},
        {"invoiceId": "INV-C-001", "vendor": "Freight & Co", "fields": {"invoiceNumber": "FC-1001", "invoiceDate": "01.03.2024", "currency": "EUR", "poNumber": "PO-C-900", "netTotal": 1000.0, "taxRate": 0.19, "taxTotal": 190.0, "grossTotal": 1190.0, "lineItems": [{"sku": None, "description": "Transport charges", "qty": 1, "unitPrice": 1000.0}]}, "confidence": 0.79, "rawText": "Invoice: FC-1001\nPO: PO-C-900\n2% Skonto if paid within 10 days\n..."},
        {"invoiceId": "INV-C-002", "vendor": "Freight & Co", "fields": {"invoiceNumber": "FC-1002", "invoiceDate": "10.03.2024", "currency": "EUR", "poNumber": "PO-C-900", "netTotal": 1000.0, "taxRate": 0.19, "taxTotal": 190.0, "grossTotal": 1190.0, "lineItems": [{"sku": None, "description": "Seefracht / Shipping", "qty": 1, "unitPrice": 1000.0}]}, "confidence": 0.73, "rawText": "Invoice: FC-1002\nPO: PO-C-900\nService: Seefracht\n..."},
        {"invoiceId": "INV-C-003", "vendor": "Freight & Co", "fields": {"invoiceNumber": "FC-1003", "invoiceDate": "20.03.2024", "currency": "EUR", "poNumber": "PO-C-901", "netTotal": 1020.0, "taxRate": 0.19, "taxTotal": 193.8, "grossTotal": 1213.8, "lineItems": [{"sku": None, "description": "Transport charges", "qty": 1, "unitPrice": 1020.0}]}, "confidence": 0.77, "rawText": "Invoice: FC-1003\nPO: PO-C-901\nSlight fuel surcharge applied\n..."},
        {"invoiceId": "INV-C-004", "vendor": "Freight & Co", "fields": {"invoiceNumber": "FC-1004", "invoiceDate": "28.03.2024", "currency": "EUR", "poNumber": "PO-C-901", "netTotal": 1020.0, "taxRate": 0.19, "taxTotal": 193.8, "grossTotal": 1213.8, "lineItems": [{"sku": None, "description": "Transport charges", "qty": 1, "unitPrice": 1020.0}]}, "confidence": 0.58, "rawText": "Invoice: FC-1004\nPO: PO-C-901\nNote: delivery confirmation pending\n..."}
    ]

@api_router.get("/demo/sample-corrections")
async def get_sample_corrections():
    """Get sample human corrections for demo"""
    return [
        {"invoiceId": "INV-A-001", "vendor": "Supplier GmbH", "corrections": [{"field": "serviceDate", "from": None, "to": "2024-01-01", "reason": "Leistungsdatum found in rawText"}], "finalDecision": "approved"},
        {"invoiceId": "INV-A-003", "vendor": "Supplier GmbH", "corrections": [{"field": "poNumber", "from": None, "to": "PO-A-051", "reason": "Only matching PO for vendor within 30 days and matching item WIDGET-002"}, {"field": "serviceDate", "from": None, "to": "2024-01-20", "reason": "Leistungsdatum found in rawText"}], "finalDecision": "approved"},
        {"invoiceId": "INV-B-001", "vendor": "Parts AG", "corrections": [{"field": "grossTotal", "from": 2400.0, "to": 2380.0, "reason": "Raw text indicates totals already include VAT; extractor overestimated"}, {"field": "taxTotal", "from": 400.0, "to": 380.0, "reason": "Recalculated from grossTotal and taxRate"}], "finalDecision": "approved"},
        {"invoiceId": "INV-B-003", "vendor": "Parts AG", "corrections": [{"field": "currency", "from": None, "to": "EUR", "reason": "Currency appears in rawText"}], "finalDecision": "approved"},
        {"invoiceId": "INV-C-001", "vendor": "Freight & Co", "corrections": [{"field": "discountTerms", "from": None, "to": "2% Skonto within 10 days", "reason": "Skonto terms in rawText"}], "finalDecision": "approved"},
        {"invoiceId": "INV-C-002", "vendor": "Freight & Co", "corrections": [{"field": "lineItems[0].sku", "from": None, "to": "FREIGHT", "reason": "Vendor uses descriptions (Seefracht/Shipping) that map to FREIGHT"}], "finalDecision": "approved"}
    ]

@api_router.get("/reference/purchase-orders")
async def get_purchase_orders():
    """Get all purchase orders"""
    pos = await db.purchase_orders.find({}, {"_id": 0}).to_list(100)
    return pos

@api_router.get("/reference/delivery-notes")
async def get_delivery_notes():
    """Get all delivery notes"""
    dns = await db.delivery_notes.find({}, {"_id": 0}).to_list(100)
    return dns

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
