package main

import "k8s.io/klog/v2"

func must(err error) {
	if err != nil {
		klog.Errorf("Must failed: %+v", err)
		panic(err)
	}
}

func must1[T any](v T, err error) T {
	must(err)
	return v
}

func must2[T1, T2 any](v1 T1, v2 T2, err error) (T1, T2) {
	must(err)
	return v1, v2
}
