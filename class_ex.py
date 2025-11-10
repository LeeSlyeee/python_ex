#클래스 명을 지정
class Calculator: 
    # 생성자 메서드
    def __init__(self): 
        # 행성자가 호출될 때마다, 해당 객체(self)의 상태를 저장하는 인스턴스 변수생성 및 초깃값 지정
        self.result = 0 

    
    # 클래스에서 사용될 메서드 생성 및 매개변수 지정 
    def add(self, num): 
        # 현재 객체의 result값에 입력받은 숫자 num을 더하고 그 결과를 result 변수에 누적
        self.result += num 
        # 함수 실행 후 결과값 result를 return 함
        return self.result
    
    def sub(self, num):
        self.result -= num
        return self.result
    
    def mul(self, num):
        self.result *= num
        return self.result
    
    def div(self, num):
        self.result /= num
        return self.result

# 클래스로부터 인스턴스를 생성하여 변수 cal1에 할당 / 할당이 되는 순간 cal1 내부의 result 값은 0으로 초기화됨
cal1 = Calculator() 

# 클래스로부터 인스턴스를 생성하여 변수 cal2에 할당 / 할당이 되는 순간 cal2 내부의 result 값은 0으로 초기화됨
cal2 = Calculator()

print(f"{cal1.add(3)} \n{cal1.add(4)} \n{cal1.sub(2)} \n{cal1.div(2)} \n{cal1.mul(8)}")
print("--------------------------------------------------")
print(f"{cal2.add(8)} \n{cal2.add(2)} \n{cal2.sub(1)} \n{cal2.mul(2)} \n{cal2.div(3)}")


