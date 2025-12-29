-- 비플페이 데이터베이스 DDL
-- 작명 규칙: Snake Case 사용
-- DBMS: MySQL / MariaDB 호환
CREATE DATABASE bple_pay;

USE bple_pay;

-- 1. USERS (사용자)
CREATE TABLE USERS (
    user_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '사용자 고유 번호',
    user_name VARCHAR(50) NOT NULL COMMENT '실명',
    phone_number VARCHAR(20) NOT NULL UNIQUE COMMENT '휴대폰 번호(Unique)',
    email VARCHAR(100) COMMENT '이메일',
    ci_di_info VARCHAR(255) COMMENT '본인인증 식별값',
    status ENUM('ACTIVE', 'SLEEP', 'WITHDRAWN') DEFAULT 'ACTIVE' COMMENT '계정 상태(ACTIVE, SLEEP, WITHDRAWN)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '가입 일시'
) COMMENT='사용자 정보';

-- 2. MERCHANTS (가맹점)
CREATE TABLE MERCHANTS (
    merchant_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '가맹점 고유 번호',
    biz_number VARCHAR(20) NOT NULL UNIQUE COMMENT '사업자 번호(Unique)',
    merchant_name VARCHAR(100) NOT NULL COMMENT '가맹점명',
    category_code VARCHAR(20) COMMENT '업종 코드',
    address VARCHAR(255) COMMENT '가맹점 주소',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '등록 일시'
) COMMENT='가맹점 정보';

-- 3. ACCOUNTS (등록 계좌)
CREATE TABLE ACCOUNTS (
    account_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '계좌 고유 번호',
    user_id INT NOT NULL COMMENT '사용자 ID',
    bank_code VARCHAR(20) NOT NULL COMMENT '은행 코드',
    account_number_enc VARCHAR(255) NOT NULL COMMENT '암호화 계좌번호',
    is_main_account BOOLEAN DEFAULT FALSE COMMENT '주 결제 계좌 여부',
    verified_at TIMESTAMP NULL COMMENT '계좌 인증 시각',
    CONSTRAINT fk_accounts_user FOREIGN KEY (user_id) REFERENCES USERS(user_id) ON DELETE CASCADE
) COMMENT='사용자 등록 계좌';

-- 4. GIFT_VOUCHER (보유 상품권)
CREATE TABLE GIFT_VOUCHER (
    voucher_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '상품권 고유 번호',
    user_id INT NOT NULL COMMENT '소유자 ID',
    voucher_name VARCHAR(100) NOT NULL COMMENT '상품권 명칭',
    total_amount DECIMAL(15, 2) NOT NULL DEFAULT 0 COMMENT '사용 가능 잔액',
    expiry_date DATE NOT NULL COMMENT '유효 기간',
    CONSTRAINT fk_voucher_user FOREIGN KEY (user_id) REFERENCES USERS(user_id) ON DELETE CASCADE
) COMMENT='사용자 보유 상품권';

-- 5. TRANSACTIONS (거래 내역)
CREATE TABLE TRANSACTIONS (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '거래 고유 번호',
    user_id INT NOT NULL COMMENT '사용자 ID',
    merchant_id INT NOT NULL COMMENT '가맹점 ID',
    payment_type ENUM('ACCOUNT', 'GIFT_CERT') NOT NULL COMMENT '결제 수단(ACCOUNT, GIFT_CERT)',
    payment_source_id INT NOT NULL COMMENT '결제 상세 참조 ID (계좌ID 또는 상품권ID)',
    amount DECIMAL(15, 2) NOT NULL COMMENT '결제 금액',
    status ENUM('COMPLETED', 'CANCELLED', 'FAILED') NOT NULL COMMENT '거래 상태(COMPLETED, CANCELLED, FAILED)',
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '거래 일시',
    CONSTRAINT fk_transactions_user FOREIGN KEY (user_id) REFERENCES USERS(user_id),
    CONSTRAINT fk_transactions_merchant FOREIGN KEY (merchant_id) REFERENCES MERCHANTS(merchant_id)
) COMMENT='거래/결제 내역';