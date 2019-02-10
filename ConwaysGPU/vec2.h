#pragma once

// Simple 2D vector utility class.

template<class T>
struct vec2 {
	vec2() = default;
	vec2(T x, T y) : x(x), y(y) { }
	vec2(const vec2<T>& v) : x(v.x), y(v.y) { }

	T x, y;

	vec2<T>& operator+=(const T& val) {
		x += val;
		y += val;
		return *this;
	}

	vec2<T>& operator+=(const vec2<T>& rhs) {
		x += rhs.x;
		y += rhs.y;
		return *this;
	}

	vec2<T>& operator-=(const vec2<T>& rhs) {
		x -= rhs.x;
		y -= rhs.y;
		return *this;
	}

	vec2<T>& operator*=(const T& scalar) {
		x *= scalar;
		y *= scalar;
		return *this;
	}
};

template<class T>
inline vec2<T> operator*(vec2<T> lhs, const T& rhs) {
	lhs *= rhs;
	return lhs;
}

template<class T>
inline vec2<T> operator/(vec2<T> lhs, const T& rhs) {
	lhs *= 1 / rhs;
	return lhs;
}

template<class T>
inline vec2<T> operator+(const vec2<T>& lhs, const vec2<T>& rhs) {
	vec2<T> vec = vec2<T>(lhs);
	vec += rhs;
	return vec;
}

template<class T>
inline vec2<T> operator+(const vec2<T>& lhs, const T rhs) {
	vec2<T> vec = vec2<T>(lhs);
	vec += rhs;
	return vec;
}

template<class T>
inline vec2<T> operator-(const vec2<T>& lhs, const vec2<T>& rhs) {
	return lhs + vec2<T>(rhs) * (-1.0f);
}

template<class T>
inline vec2<T> operator-(const vec2<T>& lhs, const T& rhs) {
	return lhs + rhs * (-1.0f);
}

template<class T>
inline bool operator==(const vec2<T>& lhs, const vec2<T>& rhs) {
	return lhs.x == rhs.x && lhs.y == rhs.y;
}