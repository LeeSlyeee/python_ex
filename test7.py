# class Bicycle(): # 클래스 선언
    
#     def __init__(self, wheel_size, color):
#         self.wheel_size = wheel_size
#         self.color = color

#     def move(self, speed):
#         print(f"자전거: 시속 {speed}킬로미터로 전진")
    
#     def turn(self, direction):
#         print(f"자전거: {direction} 회전")
    
#     def stop(self):
#         print(f"자전거({self.wheel_size}, {self.color}): 정지")
#         return f"자전거({self.wheel_size}, {self.color}): 정지"
# my_bicycle = Bicycle(22,'gray')
# my_bicycle2 = Bicycle(26,'green')
# # my_bicycle3 = Bicycle()

# # 객체 속성지정 방법
# my_bicycle2.wheel_size = 26
# my_bicycle2.color = "black"

# my_bicycle2.handle = "power"

# print(f"{my_bicycle2.stop()} 합니다. 핸들은 {my_bicycle2.handle}핸들 입니다.")



## 객체 속성 확인
# print(my_bicycle.wheel_size)
# print(my_bicycle.color)


# my_bicycle.move(20)
# my_bicycle.turn("좌상방향으로")
# my_bicycle.stop()


# my_bicycle2.move(10)
# my_bicycle2.turn("우하방향으로")
# my_bicycle2.stop()



# class Car(): 
#     instance_count = 0 # 클래스 변수 생성 및 초기화

#     def __init__(self, size, color):
#         self.size = size # 인스턴스 변수 생성 및 초기화
#         self.color = color 
#         Car.instance_count += 1 # 객체 생성 시 instance_count를 1씩 증가
#         print(f"자동차 객체의 수 {Car.instance_count}")

#     def move(self):
#         print(f"자동차 {self.color}색 {self.size}사이즈가 움직입니다.")
#         return f"자동차 {self.color}색 {self.size}사이즈가 움직입니다."

#     def turn(self):
#         print(f"'{self.move()}'가 실행됨.")

# car1 = Car("small", "red")
# car2 = Car("big", "blue")


# car1.move()
# car2.move()

# car1.turn()


# class Car():
#     instance_count = 0 # 클래스 변수 생성 및 초기화

#     # 초기화 함수(인스턴스 메서드)
#     def __init__(self, size="Small", color="Black", speed=5):
#         self.size = size # 인스턴스 변수 생성 및 초기화
#         self.color = color # 인스턴스 변수 생성 및 초기화
#         self.speed = speed
#         Car.instance_count = Car.instance_count + 1 # 클래스 변수 이용
#         print("자동차 객체의 수: {0}".format(Car.instance_count))

#     # 인스턴스 메서드
#     def move(self, speed=10):
#         self.speed = speed

#         print("자동차({0} & {1})가 ".format(self.size, self.color), end='')
#         print("시속 {0}킬로미터로 전진".format(self.speed))

#     # 인스턴스 메서드
#     def auto_cruise(self):
#         print("자율 주행 모드")
#         self.move(self.speed) # move() 함수의 인자로 인스턴스 변수를 입력


# car23 = Car()

# car23.move(120)
# car23.auto_cruise()




# ### 클래스 메서드 사용법
# class Employee:
#     # 💡 클래스 속성: 모든 직원이 공유하는 정보
#     team_name = "Development Team" 

#     def __init__(self, name, monthly_salary):
#         self.name = name
#         self.monthly_salary = monthly_salary
#         print(f"새 인스턴스 생성: {self.name}")

#     # 1. 인스턴스 메서드 (클래스 속성 접근 예시)
#     def display_info(self):
#         print(f"이름: {self.name}, 소속: {self.team_name}, 월급: {self.monthly_salary}")

#     # 2. 클래스 메서드: 쉼표로 구분된 문자열을 받아 인스턴스를 생성
#     @classmethod
#     def from_string(cls, emp_string):
#         """
#         cls는 Employee 클래스 자체를 참조합니다.
#         cls(...)를 호출하면 Employee(name, salary)와 동일합니다.
#         """

#         # 문자열을 파싱하여 데이터 추출
#         name, salary_str = emp_string.split(',')
#         # 급여 문자열을 정수형으로 변환
#         monthly_salary = int(salary_str.strip())
        
#         # 추출한 데이터를 이용해 cls(Employee)의 새로운 인스턴스를 반환
#         return cls(name, monthly_salary) 

# # --- 코드 실행 ---

# # 1. 일반 생성자(__init__)를 사용한 인스턴스 생성
# emp1 = Employee("Alice", 4000)

# # 2. 클래스 메서드(대체 생성자)를 사용한 인스턴스 생성
# data_from_file = "Bob, 5000"
# emp2 = Employee.from_string(data_from_file)

# print("\n--- 직원 정보 ---")
# emp1.display_info()
# emp2.display_info()




# 부모 클래스 정의 
class Bicycle(): 
    def __init__(self,wheel_size, color):
        self.wheel_size = wheel_size
        self.color = color

    def move(self, speed):
        print(f"자전거: 시속 {speed}킬로미터로 전진")
    
    def turn(self, direction):
        print(f"자전거: {direction}회전")

    def stop(self):
        print(f"자전거 ({self.wheel_size},{self.color}: 정지)")


# 부모 클래스에서 상속받고 기능이 추가된 클래스 정의 
class FoldingBicycle(Bicycle): 
    def __init__(self, wheel_size, color, state):
        Bicycle.__init__(self, wheel_size, color)
        # super().__init__(wheel_size, color)
        self.state = state

    def fold(self):
        self.state = "folding"
        print(f"자전거: 접기, state = {self.state}")

    def unfold(self):
        self.state = "unfolding"
        print(f"자전거: 펴기기, state = {self.state}")

    def status(self):
        print(f"자전거는 {self.state} 상태입니다.")

folding_bicycle = FoldingBicycle(27, 'white', 'unfolding')
folding_bicycle2 = FoldingBicycle(24, 'black', 'folding')


# folding_bicycle.fold()
# folding_bicycle.status()

# folding_bicycle.unfold()
# folding_bicycle.status()


from grade_manager import *

score_manager_program()