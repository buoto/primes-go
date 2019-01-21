package main

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
)

func Test_findPrimes(t *testing.T) {
	tests := []struct {
		name   string
		start  int
		end    int
		splits int
		want   []int
	}{
		{name: "small range", start: 1, end: 10, splits: 2, want: []int{1, 2, 3, 5, 7}},
		{name: "many splits", start: 1, end: 12, splits: 9, want: []int{1, 2, 3, 5, 7, 11}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findPrimes(tt.start, tt.end, tt.splits)
			sort.Ints(got) // we need to sort because order is not deterministic
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("findPrimes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_isPrime(t *testing.T) {
	tests := []struct {
		n    int
		want bool
	}{
		{1, true},
		{2, true},
		{3, true},
		{4, false},
		{5, true},
		{6, false},
		{7, true},
		{8, false},
		{9, false},
		{10, false},
		{11, true},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%d", tt.n), func(t *testing.T) {
			if got := isPrime(tt.n); got != tt.want {
				t.Errorf("isPrime(%d) = %v, want %v", tt.n, got, tt.want)
			}
		})
	}
}
