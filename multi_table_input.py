answer = input("2~9사이의 정수를 입력해주세요")
target_number = int(answer)

if 2 <= target_number < 10:
    print(f"\n--- {target_number}단 ---")
    for table_number in range(1,10):
        result = target_number * table_number
        print(f"{target_number} x {table_number} = {result}")
else:
    print("2~9사이의 정수를 입력하지 않았습니다.")
        