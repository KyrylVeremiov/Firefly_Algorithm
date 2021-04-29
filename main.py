import numpy as np

# %%
class fa:
    D = 10
    t = 20

    size_of_population=50
    alpha=1
    B0=1
    gamma=1
    A = None
    B = None
    F = None


    def initialize_population(self,a=A,b=B):
        self.A=a
        self.B=b
        return np.random.uniform(a, b, (self.size_of_population, self.D))

    def r(self,X,i,j):
        return np.linalg.norm(X[i]-X[j],2)

    def Bij(self,r):
        return self.B0/(1 + self.gamma*r*r)

    def u(self,a,b):
        return np.random.uniform(a, b, (1, self.D))

    def move_population(self,X):
        X_new= np.zeros((self.size_of_population,self.D))
        for i in range(self.size_of_population):
            for j in range(self.size_of_population):
                if self.F(X[i]>self.F(X[j])):
                    X_new[i]=X[j]+self.Bij(self.r(X,i,j))*(X[i]-X[j])+self.alpha*self.u(-1,1)
                    if X_new.any()<fa.A & X_new.any()>fa.B:
                        X_new[i]=X[i]
        return X_new

    def get_population_score(self,X):
        return self.F(X).sum()

    def find_best(self,X):
        best=min(X,key=lambda x:self.F(x))
        return (best,self.F(best))

    def solve(self,f,a=A,b=B,show_calculations=False):
        self.F=f
        self.A=a
        self.B=b
        X=self.initialize_population(a,b)
        if show_calculations:
            print('Initial population:')
            print(X)
            print('Best unit:')
            print(self.find_best(X))
            print()
        for i in range(self.t):
            X=self.move_population(X)
            if show_calculations:
                print('Score of population')
                print(self.get_population_score(X))
                print('Best unit:')
                print(self.find_best(X))
                print('----------------')
        if show_calculations:
            print('\n\nFinal population')
            print(X)
            print('\nBest unit:')
            print(self.find_best(X))
        return self.find_best(X)

def f(X):
    return abs(X * np.sin(X) + 0.1 * X).sum()
#%%
fa.solve(fa(),f,-10,10,True)

