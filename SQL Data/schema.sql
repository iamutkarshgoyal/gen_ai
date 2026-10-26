CREATE TABLE vendors (vendor_id INTEGER PRIMARY KEY, vendor_name TEXT NOT NULL, email TEXT, tax_id TEXT);
CREATE TABLE invoices (invoice_id TEXT PRIMARY KEY, vendor_id INTEGER NOT NULL REFERENCES vendors(vendor_id), 
            invoice_date DATE NOT NULL, due_date DATE NOT NULL, description TEXT NOT NULL, subtotal NUMERIC(12,2) NOT NULL, 
            tax_amount NUMERIC(12,2) NOT NULL, total_amount NUMERIC(12,2) NOT NULL, currency TEXT NOT NULL, status TEXT NOT NULL, 
            approval_status TEXT NOT NULL, department TEXT, pdf_path TEXT);
CREATE TABLE payments (payment_id TEXT PRIMARY KEY, invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id), payment_date DATE NOT NULL, 
            amount NUMERIC(12,2) NOT NULL CHECK(amount > 0), payment_method TEXT, reference_number TEXT UNIQUE);
CREATE TABLE invoice_audit (audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id), 
            action TEXT NOT NULL, old_value JSONB, new_value JSONB, approved_by TEXT, changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
