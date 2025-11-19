# 음료의 레시피를 딕셔너리로 작성

recipe = {
    "americano": [("bean", 2), ("hot_water",4)],
    "cafelatte": [("bean", 2), ("hot_milk",4)],
    "matchalatte": [("matcha", 2), ("hot_milk",4)],
    "espresso": [("bean", 4)]
}

ingredient = {
    "bean": 50,
    "matcha": 50,
    "hot_water": 100,
    "hot_milk": 100,
    "ice": 150
}


question = """
원하시는 음료의 번호를 골라주세요

1. 아메리카노
2. 카페라테
3. 말차라테
4. 에스프레소
"""

def select_bevarage():
    selected_number = int(input(f"{question}"))
    selected_bevarage = ""
    
    if selected_number == 1:
        selected_bevarage = "americano"
    elif selected_number == 2:
        selected_bevarage = "cafelatte"
    elif selected_number == 3:
        selected_bevarage = "matchalatte"
    elif selected_number == 4:
        selected_bevarage = "espresso"
        
    # return 값은 str으로 받음.
    return selected_bevarage
    

def check_recipe():
    pick_bevarage = select_bevarage()
    print(f"    >>>>>>>> {recipe[pick_bevarage]}")
    for i in recipe[pick_bevarage]:
        if i[0] in ingredient.keys():
            print(f"재료명 : {i[0]} \n{ingredient.keys()}")
        else: 
            print("틀렸어 ㅠㅠ")
    
check_recipe()