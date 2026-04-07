package humanize

import (
	"testing"
	"time"
)

func TestDuration(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		want     string
	}{
		{
			// Round to first digit after decimal point.
			name:     "milliseconds",
			duration: 123*time.Millisecond + 456*time.Microsecond,
			want:     "123.5ms",
		},
		{
			name:     "seconds with fraction",
			duration: 5*time.Second + 678*time.Millisecond,
			want:     "5.7s",
		},
		{
			// If in the range of hours, ignore the seconds.
			name:     "hours and minutes",
			duration: 2*time.Hour + 31*time.Minute + 46*time.Second,
			want:     "2h31m",
		},
		{
			name:     "minutes and seconds",
			duration: 21*time.Minute + 7*time.Second + 123*time.Millisecond,
			want:     "21m7s",
		},
		{
			name:     "microseconds",
			duration: 12*time.Microsecond + 345*time.Nanosecond,
			want:     "12.3µs",
		},
		{
			name:     "nanoseconds",
			duration: 50 * time.Nanosecond,
			want:     "50ns",
		},
		{
			// Days, ignore minutes and smaller.
			name:     "days",
			duration: 22*24*time.Hour + 3*time.Hour + 40*time.Minute,
			want:     "22d3h",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Duration(tt.duration); got != tt.want {
				t.Errorf("Duration() = %v, want %v (original: %s)", got, tt.want, tt.duration.String())
			}
		})
	}
}
