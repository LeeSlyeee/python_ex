# def ten_div(x):
#     return 10 / x

# print(ten_div(2))



# try: 
#     x = int(input("나눌 숫자를 입력하세요: "))
#     y = 10 / x
#     print(y)
# except: #예외 발생 시 실행
#     print("예외 발생")






# y = [10,20,30]

# try: 
#     index, x = map(int, input("인덱스와 나눌 숫자를 입력하세요: ").split())
#     print(y[index] / x)
# except ZeroDivisionError as e: # 숫자를 0으로 나눠서 에러가 발생 시 실행.
#     print(f"{e} : 숫자를 0으로 나눌 수 없습니다.")
# except IndexError as e: # 범위를 벗어난 인덱스에 접근하여 에러 발생 시 실행.
#     print(f"{e} : 잘못된 인덱스입니다.")






# try:
#     # 예외 발생 가능성이 있는 코드
#     file = open("example.txt", "r")
#     content = file.read()
#     file.close()

# except FileNotFoundError:
#     # 해당 예외를 처리하는 코드 
#     print("파일을 찾을 수 없습니다.")

# except PermissionError:
#     # 해당 예외를 처리하는 코드 
#     print("파일에 대한 권한이 없습니다.")

# except:
#     # 해당 예외를 처리하는 코드 
#     print("예상치 못한 예외가 발생했습니다.")






# try:
#     print(5/0)

# except:
#     print("wrong division")




# try:
#     print(5/0)
# except ZeroDivisionError as e:
#     print(e)



try: 
    print(0/1)

except Exception as e: 
    print(e)
else:
    print("no error")