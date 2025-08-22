// Package tensor provides a multi-dimensional array (tensor) implementation.
package tensor

// ConvertInt64ToInt converts a slice of int64 to a slice of int.
func ConvertInt64ToInt(s []int64) []int {
	result := make([]int, len(s))
	for i, v := range s {
		result[i] = int(v)
	}

	return result
}

// ConvertIntToInt64 converts a slice of int to a slice of int64.
func ConvertIntToInt64(s []int) []int64 {
	result := make([]int64, len(s))
	for i, v := range s {
		result[i] = int64(v)
	}

	return result
}

// Product returns the product of the elements in a slice of ints.
func Product(s []int) int {
	p := 1
	for _, v := range s {
		p *= v
	}

	return p
}
