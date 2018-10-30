package main

import (
	"sync"
)

func findPrimes(start, end, splits int) []int {
	batches := batchSizes(start, end, splits)

	c := make(chan int)

	var wg sync.WaitGroup
	wg.Add(splits)

	from, to := start, start
	// start jobs
	for _, b := range batches {
		to += b
		go func(from, to int, c chan<- int) {
			defer wg.Done()
			pushPrimesFromRange(from, to, c)
		}(from, to, c)
		from += b
	}

	// run final channel closer
	go func() {
		wg.Wait()
		close(c)
	}()

	// yield results
	var result []int
	for val := range c {
		result = append(result, val)
	}

	return result
}

func batchSizes(from, to, splits int) (sizes []int) {
	size := (to - from) / splits

	for i := 0; i < splits; i++ {
		sizes = append(sizes, size)
	}

	toAdd := (to - from) % splits
	for i := 0; i < toAdd; i++ {
		sizes[i]++
	}

	return sizes
}

func pushPrimesFromRange(from, to int, outputChan chan<- int) {
	for i := from; i < to; i++ {
		if isPrime(i) {
			outputChan <- i
		}
	}
}

func dividers(n int) (count int) {
	for i := 2; i*i <= n; i++ {
		if n%i == 0 {
			count++
		}
	}
	return count
}

func isPrime(n int) bool {
	return dividers(n) == 0
}
