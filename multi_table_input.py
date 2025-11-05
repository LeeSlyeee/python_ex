target_number = input("2~9사이의 정수를 입력해주세요")

if target_number.lower() == 'q':
        print("프로그램을 종료합니다.")

elif not target_number.isdigit(): 
    print("숫자를 입력하지 않았습니다. 프로그램을 종료합니다.")    

elif not target_number:
    print("아무것도 입력하지 않았습니다. 프로그램을 종료합니다.")

target_number = int(target_number)

if 2 <= target_number < 10:
    print(f"\n--- {target_number}단 ---")
    for table_number in range(1,10):
        result = target_number * table_number
        print(f"{target_number} x {table_number} = {result}")
else:
    print("2~9사이의 정수를 입력하지 않았습니다.")
    