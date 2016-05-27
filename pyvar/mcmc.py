class MCMC:

    def __init__(self, pnames):
        self.__npara = len(pnames)
        self.__nsim = 0
        self.__parasim = []
        self.__pnames = pnames

    def __str__(self):
        pass

    def iat(self, L):
        pass

    def seconds_per_effective_draws(self, L):
        pass

    def append(self, para_list):
        for para in para_list:
            self.__parasim.append(dict(zip(self.__pnames, para_list)))

        self.__nsim += 1

    def get(self, i):
        if i > self.__nsim:
            print("Error: i not found!")
            return

        parai = self.__parasim[i]
        return parai
    
    def nsim(self):
        return self.__nsim


if __name__ == "__main__":
    test = MCMC(("alpha", "beta"))
    test.append((0, 1))
    print(test.get(0))
