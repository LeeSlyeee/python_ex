def func_multi_table():    
    answer = input("2~9사이의 정수를 입력해주세요! q입력시 프로그램을 종료합니다.")

    if answer.lower() == 'q':
            print("프로그램을 종료합니다.")
            return False

    elif not answer.isdigit(): 
        print("숫자를 입력하지 않았습니다. 입력을 다시 시작합니다.")    
        return func_multi_table()

    elif not answer:
        print("아무것도 입력하지 않았습니다. 입력을 다시 시작합니다.")
        return func_multi_table()       
    
    else : target_number = int(answer)    
    
    
    if 2 <= target_number < 10:
        print(f"\n--- {target_number}단 ---")
        for table_number in range(1,10):
            result = target_number * table_number
            print(f"{target_number} x {table_number} = {result}")

    else:
        print("2~9사이의 정수를 입력하지 않았습니다. 입력을 다시 시작합니다.")
        return func_multi_table()       

    return False

func_multi_table()