package main

import (
	"sync"
	"encoding/json"
	"github.com/bitly/go-nsq"
	"log"
)

func main() {
	consume()
}

func produce(primes []int) {
	config := nsq.NewConfig()
	w, _ := nsq.NewProducer("127.0.0.1:4150", config)
	
		j , _ := json.Marshal(primes)
		err := w.Publish("found_prime", j)
		if err != nil {
			log.Panic("Could not connect")
		}
	w.Stop()
}

func consume() {
	wg := &sync.WaitGroup{}
  	wg.Add(1)

  	config := nsq.NewConfig()
	q, _ := nsq.NewConsumer("find_prime", "ch", config)
  	q.AddHandler(nsq.HandlerFunc(func(message *nsq.Message) error {
	  	var batch map[string]int
	  	if err := json.Unmarshal(message.Body, &batch); err != nil {
        panic(err)
		}
		pushPrimesFromRange(batch["from"], batch["to"])
		return nil
  	}))
  	err := q.ConnectToNSQD("127.0.0.1:4150")
  	if err != nil {
      log.Panic("Could not connect")
  	}
	wg.Wait()
}

func pushPrimesFromRange(from, to int) {
	primes := []int {}
	for i := from; i < to; i++ {
		if isPrime(i) {
			primes = append(primes, i)
		}
	}
	produce(primes)
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

