import random
import copy
import math
firstlist = []
maxweigths=15
weigth=[2,5,7,3,5]
price=[12,3,2,4,5]

bestlist=0
def scoreF(lists,value):#function that calculates the score
    score=0
    for j in range (0,5):
        if lists[j]==1:
                score=score+value[j]
    return score

def eval1(x):
    x=1-x
    return x

def hillclimbing(randomlist):  
    sol=[] 
    solscore=[]
    
    curentbest=-100
    curentscorebest=-100
    weigths1=0
    
    for i in range(0,5):#generate list of solutions
        mem=copy.copy(randomlist)
        if randomlist[i]==0:
            mem[i]=1
     
        elif randomlist[i]==1:
            mem[i]=0
        sol.append(mem)
    print(sol)

    for i in range(0,5):#score for generated solutions            
        solscore.append(scoreF(sol[i],price))

    score1=scoreF(randomlist,price)#score for the original solution           
    print(solscore)
    print(score1)


    for i in range(0,5):#calculate the total weigths of generated solutions
        weigths=0
        weigths=scoreF(sol[i],weigth)
        if weigths>maxweigths:
            solscore[i]=-10000000000
        

    weigths1=scoreF(randomlist,weigth)          
    if weigths1>maxweigths:#check if the generated solution is within weight limits
        score1=-10000000000
    print(solscore)    
    
    for i in range(0,5):#calculate the best solution
        if solscore[i]>score1:
            curentbest=solscore[i]
            curentscorebest=i
            score1=solscore[i]
                    
    print("Best solution:")
    if curentbest==-100:
        print(randomlist)
        curentbest=scoreF(randomlist,price)
        best=randomlist
    elif curentbest>-100:
        print(sol[curentscorebest])
        best=sol[curentscorebest]
    return best , curentbest

def it(solution,bestlist):
    while solution[1]>bestlist:        
            bestlist=solution[1]
            solution=hillclimbing(solution[0])
    
    return solution 

for i in range(0,5):#generate an initial solution
    n = random.randint(0,1)
    firstlist.append(n)
print(firstlist)

solution=hillclimbing(firstlist)

   
solution=it(solution,bestlist)
if random.random()<math.exp((eval1(solution[1])-eval1(bestlist))/maxweigths):#add probability for simulated
            #bestlist=solution[1]
        print("--------------------------------------------------------------------------------->")
        solution=it(solution, solution[1])
print(random.random())
print(math.exp((eval1(solution[1])-eval1(bestlist))/maxweigths))
print(eval1(solution[1]),"and ",eval1(bestlist))   

    
print(scoreF(solution[0],price))