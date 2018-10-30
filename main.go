package main

import (
	"fmt"
)

func main() {
	start, end, splits := 1, 12, 9

	primes := findPrimes(start, end, splits)
	fmt.Printf("Found primes: %v", primes)
}
