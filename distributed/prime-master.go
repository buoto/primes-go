package main

import (
	"sync"
	"encoding/json"
	"github.com/bitly/go-nsq"
	"log"
)

func findPrimesMaster(start, end, splits int) []int {
	batches := batchSizes(start, end, splits)
	connections := produce(start, start, batches)
	return consume(connections)
}

func produce(from, to int, batches []int) int {
	config := nsq.NewConfig()
	w, _ := nsq.NewProducer("127.0.0.1:4150", config)
	connections := 0
	for _, b := range batches {
		connections++
		to += b
		m := map[string]int{"from": from, "to": to}
		j , _ := json.Marshal(m)
		err := w.Publish("find_prime", j)
		if err != nil {
			log.Panic("Could not connect")
		}
		from += b
	}
	w.Stop()
	return connections
}



func consume(connections int) []int {
	wg := &sync.WaitGroup{}
  	wg.Add(1)

  	config := nsq.NewConfig()
	q, _ := nsq.NewConsumer("found_prime", "ch", config)
	var result []int
  	q.AddHandler(nsq.HandlerFunc(func(message *nsq.Message) error {
	  	primes := []int {}
	  	if err := json.Unmarshal(message.Body, &primes); err != nil {
        panic(err)
    	}
		log.Printf("Got a message: %v", message)
	  	connections--
	  	if(connections == 0) {
			wg.Done()
		}
		result = append(result, primes...)
      	return nil
  	}))
  	err := q.ConnectToNSQD("127.0.0.1:4150")
  	if err != nil {
      log.Panic("Could not connect")
  	}
	wg.Wait()
	return result 
}
