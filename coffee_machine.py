class CoffeeMachine:
    def __init__(self):
        # 재료 수량 초기화
        self.resources = {
            "원두": 300,
            "따듯한 물": 400,
            "녹차": 100,
            "얼음": 300,
            "따듯한 우유": 200  # 레시피에 우유가 있어서 추가했습니다.
        }
        
        # 레시피 정의
        self.recipes = {
            "아메리카노": {"원두": 50, "따듯한 물": 100},
            "카페라떼": {"원두": 50, "따듯한 우유": 100},
            "말차라떼": {"녹차": 50, "따듯한 우유": 100},
            "에스프레소": {"원두": 100}
        }
        
        # 옵션 비용 (재료 소모)
        self.ice_cost = {"얼음": 50}

    def show_menu(self):
        print("\n" + "="*20 + " 메뉴 " + "="*20)
        for i, name in enumerate(self.recipes, 1):
            print(f"{i}. {name}")
        print("="*46)

    def check_resources(self, ingredients):
        """주문한 커피의 재료가 충분한지 확인합니다."""
        for item, amount in ingredients.items():
            if self.resources.get(item, 0) < amount:
                print(f"*** 죄송합니다. {item} 재료가 부족합니다. ㅠㅠ")
                return False
        return True

    def deduct_resources(self, ingredients):
        """재료를 사용합니다."""
        for item, amount in ingredients.items():
            self.resources[item] -= amount

    def process_order(self):
        self.show_menu()
        try:
            choice_num = int(input("커피를 선택해주세요 (번호 입력): "))
            menu_names = list(self.recipes.keys())
            
            if 1 <= choice_num <= len(menu_names):
                choice = menu_names[choice_num - 1]
            else:
                print("잘못된 번호입니다.")
                return
        except ValueError:
            print("숫자를 입력해주세요.")
            return

        # 1. 선택한 커피의 기본 재료 확인
        ingredients = self.recipes[choice]
        if not self.check_resources(ingredients):
            return

        # 2. 온/냉 음료 선택
        is_ice = False
        # 에스프레소는 보통 뜨겁게 마시지만, 로직상 물어볼 수 있음. 여기서는 흐름도대로 진행.
        temp_choice = input("따뜻한 거(H) / 차가운 거(I) 선택해주세요: ").upper()
        
        final_ingredients = ingredients.copy()
        
        if temp_choice == 'I':
            is_ice = True
            # 얼음 재료 확인
            if not self.check_resources(self.ice_cost):
                return
            # 얼음 재료 추가
            final_ingredients.update(self.ice_cost)

        # 3. 커피 제작 (재료 차감)
        print(f"\n{choice} 제조 중...")
        self.deduct_resources(final_ingredients)
        
        # 4. 제공
        temp_str = "아이스" if is_ice else "따뜻한"
        print(f"*** {temp_str} {choice}가 준비되었습니다. 맛있게 드세요. ^^")
        
        # 남은 재료 확인 (디버깅용)
        # print(f"(남은 재료: {self.resources})")

# --- 실행 코드 ---
if __name__ == "__main__":
    machine = CoffeeMachine()
    
    while True:
        machine.process_order()
        cont = input("\n계속 주문하시겠습니까? (y/n): ")
        if cont.lower() != 'y':
            break
