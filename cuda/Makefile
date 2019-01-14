EXEC=primes

all: primes

run: all
	./$(EXEC)

primes: main.cu
	nvcc main.cu -o $(EXEC)
