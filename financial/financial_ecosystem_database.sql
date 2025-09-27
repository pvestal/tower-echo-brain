-- Echo Brain Financial Ecosystem Database
-- Comprehensive schema for individual, business, trust, and family finances

-- ================================
-- FINANCIAL ENTITIES
-- ================================

CREATE TABLE financial_entities (
    entity_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('individual', 'business', 'trust', 'family', 'board', 'estate')),
    entity_name VARCHAR(255) NOT NULL,
    legal_name VARCHAR(255),
    tax_id VARCHAR(50),  -- SSN for individuals, EIN for businesses
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- ================================
-- PLAID CONNECTIONS
-- ================================

CREATE TABLE plaid_connections (
    connection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id) ON DELETE CASCADE,
    item_id VARCHAR(255) NOT NULL,  -- Plaid item ID
    access_token TEXT NOT NULL,  -- Encrypted
    institution_id VARCHAR(100),
    institution_name VARCHAR(255),
    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_sync TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    error_code VARCHAR(50),
    metadata JSONB DEFAULT '{}'
);

-- Index for quick lookups
CREATE INDEX idx_plaid_entity ON plaid_connections(entity_id, is_active);

-- ================================
-- BANK ACCOUNTS (via Plaid)
-- ================================

CREATE TABLE bank_accounts (
    account_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plaid_account_id VARCHAR(255) UNIQUE NOT NULL,
    connection_id UUID NOT NULL REFERENCES plaid_connections(connection_id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id),
    account_name VARCHAR(255),
    account_type VARCHAR(50),  -- checking, savings, investment, credit, loan
    account_subtype VARCHAR(50),
    current_balance DECIMAL(20, 2),
    available_balance DECIMAL(20, 2),
    credit_limit DECIMAL(20, 2),
    currency_code VARCHAR(3) DEFAULT 'USD',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

-- Index for balance queries
CREATE INDEX idx_accounts_entity ON bank_accounts(entity_id, account_type);
CREATE INDEX idx_accounts_balance ON bank_accounts(entity_id, current_balance);

-- ================================
-- TRANSACTIONS (from Plaid)
-- ================================

CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plaid_transaction_id VARCHAR(255) UNIQUE NOT NULL,
    account_id UUID NOT NULL REFERENCES bank_accounts(account_id) ON DELETE CASCADE,
    amount DECIMAL(20, 2) NOT NULL,
    date DATE NOT NULL,
    name VARCHAR(500),
    merchant_name VARCHAR(255),
    category TEXT[],
    pending BOOLEAN DEFAULT FALSE,
    transaction_type VARCHAR(50),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for transaction queries
CREATE INDEX idx_transactions_account_date ON transactions(account_id, date DESC);
CREATE INDEX idx_transactions_category ON transactions USING gin(category);

-- ================================
-- BUSINESS ENTITIES
-- ================================

CREATE TABLE businesses (
    business_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id) ON DELETE CASCADE,
    business_name VARCHAR(255) NOT NULL,
    business_type VARCHAR(50),  -- LLC, Corp, Partnership, etc
    ein VARCHAR(20),
    state_of_formation VARCHAR(2),
    formation_date DATE,
    fiscal_year_end VARCHAR(5),  -- MM-DD
    industry_code VARCHAR(10),
    annual_revenue DECIMAL(20, 2),
    annual_expenses DECIMAL(20, 2),
    employee_count INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- ================================
-- OWNERSHIP STRUCTURE
-- ================================

CREATE TABLE ownership (
    ownership_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    business_id UUID NOT NULL REFERENCES businesses(business_id) ON DELETE CASCADE,
    owner_entity_id UUID NOT NULL REFERENCES financial_entities(entity_id),
    ownership_percentage DECIMAL(5, 2) CHECK (ownership_percentage >= 0 AND ownership_percentage <= 100),
    ownership_type VARCHAR(50),  -- common, preferred, etc
    acquired_date DATE,
    vesting_schedule JSONB,
    is_active BOOLEAN DEFAULT TRUE
);

-- Ensure ownership doesn't exceed 100%
CREATE OR REPLACE FUNCTION check_ownership_total()
RETURNS TRIGGER AS $$
BEGIN
    IF (
        SELECT SUM(ownership_percentage)
        FROM ownership
        WHERE business_id = NEW.business_id
        AND is_active = TRUE
    ) > 100 THEN
        RAISE EXCEPTION 'Total ownership cannot exceed 100%%';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER check_ownership_total_trigger
AFTER INSERT OR UPDATE ON ownership
FOR EACH ROW EXECUTE FUNCTION check_ownership_total();

-- ================================
-- TRUSTS (Vestal Estate)
-- ================================

CREATE TABLE trusts (
    trust_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id) ON DELETE CASCADE,
    trust_name VARCHAR(255) NOT NULL,
    trust_type VARCHAR(50),  -- revocable, irrevocable, charitable, etc
    trustee_entity_id UUID REFERENCES financial_entities(entity_id),
    successor_trustee_id UUID REFERENCES financial_entities(entity_id),
    created_date DATE,
    trust_document_url TEXT,  -- Encrypted link to document
    distribution_rules JSONB,
    investment_policy JSONB,
    is_active BOOLEAN DEFAULT TRUE
);

-- ================================
-- TRUST BENEFICIARIES
-- ================================

CREATE TABLE trust_beneficiaries (
    beneficiary_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trust_id UUID NOT NULL REFERENCES trusts(trust_id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id),
    beneficiary_type VARCHAR(50),  -- primary, contingent
    distribution_percentage DECIMAL(5, 2),
    distribution_rules JSONB,  -- Age limits, conditions, etc
    vesting_date DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- ================================
-- TRUST ASSETS
-- ================================

CREATE TABLE trust_assets (
    asset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trust_id UUID NOT NULL REFERENCES trusts(trust_id) ON DELETE CASCADE,
    asset_type VARCHAR(50),  -- cash, securities, real_estate, business_interest
    description TEXT,
    current_value DECIMAL(20, 2),
    acquisition_date DATE,
    acquisition_value DECIMAL(20, 2),
    location VARCHAR(500),  -- For real estate
    account_id UUID REFERENCES bank_accounts(account_id),  -- If it's a bank account
    metadata JSONB DEFAULT '{}'
);

-- ================================
-- BOARD OF DIRECTORS
-- ================================

CREATE TABLE board_of_directors (
    board_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    board_name VARCHAR(255) NOT NULL,
    entity_id UUID REFERENCES financial_entities(entity_id),  -- Associated entity
    quorum_requirement INTEGER DEFAULT 2,
    major_decision_threshold DECIMAL(3, 2) DEFAULT 0.75,  -- 75%
    minor_decision_threshold DECIMAL(3, 2) DEFAULT 0.51,  -- 51%
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE board_members (
    member_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    board_id UUID NOT NULL REFERENCES board_of_directors(board_id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id),
    role VARCHAR(50),  -- chair, secretary, treasurer, member
    voting_weight DECIMAL(3, 2) DEFAULT 1.0,
    joined_date DATE DEFAULT CURRENT_DATE,
    term_end_date DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- ================================
-- BOARD DECISIONS
-- ================================

CREATE TABLE board_decisions (
    decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    board_id UUID NOT NULL REFERENCES board_of_directors(board_id),
    submitted_by UUID NOT NULL REFERENCES financial_entities(entity_id),
    decision_type VARCHAR(50),  -- trust_distribution, investment, loan, policy
    title VARCHAR(500),
    description TEXT,
    amount DECIMAL(20, 2),
    voting_threshold DECIMAL(3, 2),
    status VARCHAR(20) DEFAULT 'pending',  -- pending, approved, rejected, executed
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMP,
    executed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE board_votes (
    vote_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id UUID NOT NULL REFERENCES board_decisions(decision_id) ON DELETE CASCADE,
    member_id UUID NOT NULL REFERENCES board_members(member_id),
    vote VARCHAR(10) CHECK (vote IN ('yes', 'no', 'abstain')),
    vote_reason TEXT,
    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(decision_id, member_id)
);

-- ================================
-- LOAN APPLICATIONS (loan-search)
-- ================================

CREATE TABLE loan_applications (
    application_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id),
    loan_type VARCHAR(50),  -- personal, mortgage, business, auto
    requested_amount DECIMAL(20, 2),
    approved_amount DECIMAL(20, 2),
    lender_name VARCHAR(255),
    interest_rate DECIMAL(5, 2),
    term_months INTEGER,
    status VARCHAR(50),  -- searching, applied, approved, rejected, funded
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decision_at TIMESTAMP,
    funded_at TIMESTAMP,
    credit_score INTEGER,
    debt_to_income DECIMAL(5, 2),
    collateral JSONB,
    metadata JSONB DEFAULT '{}'
);

-- ================================
-- TRUST DISTRIBUTIONS
-- ================================

CREATE TABLE trust_distributions (
    distribution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trust_id UUID NOT NULL REFERENCES trusts(trust_id),
    beneficiary_id UUID NOT NULL REFERENCES trust_beneficiaries(beneficiary_id),
    amount DECIMAL(20, 2) NOT NULL,
    distribution_type VARCHAR(50),  -- regular, emergency, education, special
    purpose TEXT,
    approved_by UUID REFERENCES financial_entities(entity_id),
    board_decision_id UUID REFERENCES board_decisions(decision_id),
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approved_at TIMESTAMP,
    distributed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    documentation JSONB
);

-- ================================
-- INVESTMENT PORTFOLIOS
-- ================================

CREATE TABLE investment_portfolios (
    portfolio_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id),
    portfolio_name VARCHAR(255),
    portfolio_type VARCHAR(50),  -- individual, trust, business
    custodian VARCHAR(255),
    account_number VARCHAR(100),
    total_value DECIMAL(20, 2),
    cash_balance DECIMAL(20, 2),
    last_rebalanced DATE,
    target_allocation JSONB,
    actual_allocation JSONB,
    risk_score INTEGER CHECK (risk_score >= 1 AND risk_score <= 10)
);

-- ================================
-- FINANCIAL GOALS
-- ================================

CREATE TABLE financial_goals (
    goal_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES financial_entities(entity_id),
    goal_name VARCHAR(255),
    goal_type VARCHAR(50),  -- retirement, education, purchase, emergency_fund
    target_amount DECIMAL(20, 2),
    current_amount DECIMAL(20, 2),
    target_date DATE,
    monthly_contribution DECIMAL(20, 2),
    priority INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

-- ================================
-- AGGREGATE VIEWS
-- ================================

-- Family Net Worth View
CREATE OR REPLACE VIEW family_net_worth AS
SELECT
    fe.entity_id,
    fe.entity_name,
    fe.entity_type,
    COALESCE(
        (SELECT SUM(current_balance)
         FROM bank_accounts ba
         WHERE ba.entity_id = fe.entity_id
         AND ba.account_type NOT IN ('credit', 'loan')
         AND ba.is_active = TRUE),
        0
    ) as liquid_assets,
    COALESCE(
        (SELECT SUM(ta.current_value)
         FROM trust_assets ta
         JOIN trusts t ON ta.trust_id = t.trust_id
         WHERE t.trustee_entity_id = fe.entity_id),
        0
    ) as trust_assets,
    COALESCE(
        (SELECT SUM(b.annual_revenue - b.annual_expenses) * 3  -- 3x multiplier
         FROM businesses b
         JOIN ownership o ON b.business_id = o.business_id
         WHERE o.owner_entity_id = fe.entity_id
         AND o.is_active = TRUE),
        0
    ) as business_value
FROM financial_entities fe
WHERE fe.entity_type IN ('individual', 'trust', 'business');

-- Consolidated Family View
CREATE OR REPLACE VIEW family_financial_power AS
SELECT
    SUM(liquid_assets) as total_liquid,
    SUM(trust_assets) as total_trust,
    SUM(business_value) as total_business,
    SUM(liquid_assets + trust_assets + business_value) as total_net_worth,
    CASE
        WHEN SUM(liquid_assets + trust_assets + business_value) > 10000000 THEN 'Ultra High Net Worth'
        WHEN SUM(liquid_assets + trust_assets + business_value) > 1000000 THEN 'High Net Worth'
        WHEN SUM(liquid_assets + trust_assets + business_value) > 500000 THEN 'Affluent'
        WHEN SUM(liquid_assets + trust_assets + business_value) > 100000 THEN 'Mass Affluent'
        ELSE 'Building Wealth'
    END as wealth_tier
FROM family_net_worth;

-- ================================
-- AUDIT TRIGGERS
-- ================================

-- Audit all financial changes
CREATE TABLE financial_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100),
    operation VARCHAR(10),
    entity_id UUID,
    user_id UUID,
    changed_data JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create audit function
CREATE OR REPLACE FUNCTION audit_financial_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO financial_audit_log (table_name, operation, entity_id, changed_data)
    VALUES (
        TG_TABLE_NAME,
        TG_OP,
        COALESCE(NEW.entity_id, OLD.entity_id),
        jsonb_build_object(
            'old', to_jsonb(OLD),
            'new', to_jsonb(NEW)
        )
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers to sensitive tables
CREATE TRIGGER audit_bank_accounts AFTER INSERT OR UPDATE OR DELETE ON bank_accounts
    FOR EACH ROW EXECUTE FUNCTION audit_financial_changes();

CREATE TRIGGER audit_trust_distributions AFTER INSERT OR UPDATE OR DELETE ON trust_distributions
    FOR EACH ROW EXECUTE FUNCTION audit_financial_changes();

CREATE TRIGGER audit_board_decisions AFTER INSERT OR UPDATE OR DELETE ON board_decisions
    FOR EACH ROW EXECUTE FUNCTION audit_financial_changes();

-- ================================
-- PERMISSIONS & SECURITY
-- ================================

-- Grant appropriate permissions
GRANT SELECT ON family_net_worth TO echo_read;
GRANT SELECT ON family_financial_power TO echo_read;
GRANT ALL ON ALL TABLES IN SCHEMA public TO echo_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO echo_admin;