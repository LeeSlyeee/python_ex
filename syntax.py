answer_age = input("나이를 적어주세요")
if not answer_age.isalpha(): 
    int_age = int(answer_age)   
    status = "성인" if int_age >= 18 else "미성년"
    print(status) 
        
else:
    print("숫자로만 적어주세요")
    