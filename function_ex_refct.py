def func_multi_table():
    """
    사용자로부터 2~9 사이의 정수를 입력받아 해당 단의 구구단을 출력합니다.
    'q'를 입력하면 프로그램을 종료합니다.
    """
    print("✨ 구구단 출력 프로그램 시작 ✨")
    
    while True:
        # 사용자 입력 받기
        answer = input("\n2~9 사이의 정수를 입력해주세요! ('q' 입력시 종료): ").strip()
        
        # 1. 종료 조건 확인 ('q' 또는 'Q')
        if answer.lower() == 'q':
            print("프로그램을 종료합니다.")
            break  # 루프 종료
            
        # 2. 아무것도 입력하지 않은 경우 (공백만 입력한 경우도 strip()으로 처리됨)
        if not answer:
            print("❌ 아무것도 입력하지 않았습니다. 다시 입력해주세요.")
            continue # 루프 처음으로 돌아가 다시 입력받기
            
        # 3. 숫자인지 확인
        if not answer.isdigit():
            print("❌ 숫자가 아닌 다른 문자를 입력했습니다. 다시 입력해주세요.")
            continue
            
        # 4. 숫자로 변환 후 범위 확인
        target_number = int(answer)
        
        if 2 <= target_number <= 9:
            # 유효한 입력 (2~9)
            print(f"\n--- 🌟 {target_number}단 🌟 ---")
            for table_number in range(1, 10):
                result = target_number * table_number
                # f-string을 사용하여 정렬 및 가독성 향상
                print(f"{target_number} x {table_number} = {result:2}")
                
            # 구구단 출력 후 다시 입력받기 위해 루프 계속
            
        else:
            # 2~9 범위 밖의 숫자
            print("❌ 2~9 사이의 정수가 아닙니다. 다시 입력해주세요.")
            continue # 루프 처음으로 돌아가 다시 입력받기

func_multi_table()