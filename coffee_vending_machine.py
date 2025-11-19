recipe = {
    "americano": [("bean", 25), ("hot_water", 8)],
    "cafelatte": [("bean", 25), ("hot_milk", 8)],
    "matchalatte": [("matcha", 25), ("hot_milk", 8)],
    "espresso": [("bean", 30)]
} # 음료별 레시피 딕셔너리

ingredient = {
    "bean": 50,
    "matcha": 50,
    "hot_water": 100,
    "hot_milk": 100,
    "ice": 150
} # 재료별 초기 재고 딕셔너리

price = {
    "americano": 2000,
    "cafelatte": 2500,
    "matchalatte": 3000,
    "espresso": 1500
} # 음료별 가격 딕셔너리

question = """
원하시는 음료의 번호를 골라주세요

1. 아메리카노
2. 카페라테
3. 말차라테
4. 에스프레소
""" # 음료 선택 질문 문자열

def select_bevarage(): # 음료 선택 함수
    selected_number = int(input(f"{question}")) # input 값은 str으로 받음.
    selected_bevarage = "" # 빈 문자열 생성
    
    if selected_number == 1: # 아메리카노 선택 시
        selected_bevarage = "americano" # 선택한 음료를 문자열로 저장
    elif selected_number == 2: # 카페라테 선택 시
        selected_bevarage = "cafelatte" # 선택한 음료를 문자열로 저장
    elif selected_number == 3: # 말차라테 선택 시
        selected_bevarage = "matchalatte" # 선택한 음료를 문자열로 저장
    elif selected_number == 4: # 에스프레소 선택 시 
        selected_bevarage = "espresso" # 선택한 음료를 문자열로 저장
        
    # return 값은 str으로 받음.
    return selected_bevarage # 선택한 음료 문자열 반환
    

def print_price(): # 가격 출력 함수
    pick_bevarage = select_bevarage() # 음료 선택 함수 호출
    print(f"{pick_bevarage}의 가격은 {price[pick_bevarage]}원 입니다.") # 선택한 음료의 가격 출력   
# print_price() # 가격 출력 함수 호출

def pay_money(): # 결제 함수
    pick_bevarage = select_bevarage() # 음료 선택 함수 호출
    paid_amount = int(input(f"{pick_bevarage}의 가격은 {price[pick_bevarage]}원 입니다. 금액을 투입해주세요: ")) # 결제 금액 입력 받음.
    
    if paid_amount < price[pick_bevarage]: # 결제 금액이 음료 가격보다 적을 시
        print("금액이 부족합니다. 다시 투입해주세요.") # 금액 부족 메시지 출력
        return pay_money() # 재귀 호출로 다시 결제 함수 실행
    
    else: # 결제 금액이 음료 가격보다 많거나 같을 시
        change = paid_amount - price[pick_bevarage] # 잔돈 계산
        print(f"결제가 완료되었습니다.\n잔돈은 {change}원 입니다.") # 결제 완료 및 잔돈 출력
        return pick_bevarage # 선택한 음료 문자열 반환
# pay_money() # 결제 함수 호출
    

def make_bevarage(): # 음료 제조 함수
    pick_bevarage = pay_money() # 결제 함수 호출
    for ingredient_name, required_amount in recipe[pick_bevarage]: # 선택한 음료의 레시피 순회
        if ingredient[ingredient_name] < required_amount: # 재고가 부족할 시
            print(f"{ingredient_name} 재고가 부족합니다. 다른 음료를 선택해주세요.") # 재고 부족 메시지 출력
            return make_bevarage() # 재귀 호출로 다시 음료 제조 함수 실행
    for ingredient_name, required_amount in recipe[pick_bevarage]: # 선택한 음료의 레시피 순회
        ingredient[ingredient_name] -= required_amount # 재고에서 사용한 재료만큼 차감
    print(f"{pick_bevarage}가 완성되었습니다. 맛있게 드세요!") # 음료 완성 메시지 출력
# make_bevarage() # 음료 제조 함수 호출

def check_ingredient(): # 재고 확인 함수
    print("현재 재고 현황:") # 재고 현황 메시지 출력
    for ingredient_name, amount in ingredient.items(): # 재고 딕셔너리 순회
        print(f"{ingredient_name}: {amount}g") # 재고 출력
# check_ingredient() # 재고 확인 함수 호출

def coffee_vending_machine(): # 커피 자판기 함수
    while True: # 무한 루프
        make_bevarage() # 음료 제조 함수 호출
        check_ingredient() # 재고 확인 함수 호출
        continue_choice = input("계속 하시겠습니까? (y/n): ") # 계속 여부 입력 받음.
        if continue_choice.lower() != 'y': # 계속하지 않을 시
            print("이용해주셔서 감사합니다!") # 종료 메시지 출력
            break # 루프 종료
        
        
coffee_vending_machine() # 커피 자판기 함수 호출

