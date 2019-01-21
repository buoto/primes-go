package main

import (
	"fmt"
)

func main() {
	start, end, splits := 1, 12, 2

	primes := findPrimesMaster(start, end, splits)
	fmt.Printf("Found primes: %v", primes)
}