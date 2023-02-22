#pragma once


template <typename T, typename POINT>
class bbox2 {
public:
    bbox2() {
        clear();
    }

    bbox2(const POINT &min, const POINT &max)
    {
        min_ = min;
        max_ = max;
    }

    bbox2(T x, T y, T w, T h)    { min_.x = x;    min_.y = y;    max_.x = x + w;  max_.y = y + h;  }

    explicit bbox2(const T *v)    { min_.x = v[0];  min_.y = v[1];  max_.x = v[2];  max_.y = v[3];  }

    template<typename TT>
    bbox2(const bbox2<TT, POINT> &rhs)    { min_ = POINT(rhs.min()); max_ = POINT(rhs.max()); }

    void clear()
    {
        min_.x = std::numeric_limits<T>::max();
        min_.y = std::numeric_limits<T>::max();

        max_.x = std::numeric_limits<T>::lowest();
        max_.y = std::numeric_limits<T>::lowest();
    }

    void grow(const POINT &pt)
    {
        min_.x = std::min(min_.x, pt.x);
        min_.y = std::min(min_.y, pt.y);

        max_.x = std::max(max_.x, pt.x);
        max_.y = std::max(max_.y, pt.y);
    }

    void grow(const bbox2<T, POINT> &bbox)
    {
        if (bbox.empty()) return;

        min_.x = std::min(min_.x, bbox.min_.x);
        min_.y = std::min(min_.y, bbox.min_.y);

        max_.x = std::max(max_.x, bbox.max_.x);
        max_.y = std::max(max_.y, bbox.max_.y);
    }

    void clip(const bbox2<T, POINT> &bbox)
    {
        min_.x = std::max(min_.x, bbox.min_.x);
        min_.y = std::max(min_.y, bbox.min_.y);

        max_.x = std::min(max_.x, bbox.max_.x);
        max_.y = std::min(max_.y, bbox.max_.y);
    }

    bool empty() const
    {
        if (max_.x < min_.x) return true;
        if (max_.y < min_.y) return true;
        return false;
    }

    bool contains(const POINT &pt) const
    {
        if (pt.x < min_.x || pt.y < min_.y) return false;
        if (pt.x > max_.x || pt.y > max_.y) return false;
        return true;
    }

    bool contains(const bbox2<T, POINT> &box) const
    {
        if (box.min_.x < min_.x || box.min_.y < min_.y) return false;
        if (box.max_.x > max_.x || box.max_.y > max_.y) return false;
        return true;
    }

    bool intersects(const bbox2<T, POINT> &box) const
    {
        if (min_.x > box.max_.x || max_.x < box.min_.x) return false;
        if (min_.y > box.max_.y || max_.y < box.min_.y) return false;
        return true;
    }

    POINT center() const
    {
        return (max_ + min_) / (T) 2;
    }

    POINT size() const
    {
        return max_ - min_;
    }

    T  area() const  { return (max_.x - min_.x) * (max_.y - min_.y); }
    T  width() const  { return max_.x - min_.x; }
    T  height() const  { return max_.y - min_.y; }

    const POINT &min() const
    {
        return min_;
    }

    const POINT &max() const
    {
        return max_;
    }

    POINT &min()
    {
        return min_;
    }

    POINT &max()
    {
        return max_;
    }

    T distance2(const POINT &pt) const
    {
        T dist2 = 0;
        T aux;
        for (int k = 0; k < 2; ++k) {
            if ( (aux = (pt[k]-min_[k]))<0. )
                dist2 += aux*aux;
            else if ( (aux = (max_[k]-pt[k]))<0. )
                dist2 += aux*aux;
        }
        return dist2;
    }

    bbox2& operator += (const POINT& rhs) {
        min_ += rhs;
        max_ += rhs;
        return *this;
    }

    bbox2& operator -= (const POINT& rhs) {
        min_ -= rhs;
        max_ -= rhs;
        return *this;
    }

    bool operator == (const bbox2& rhs) const { return min() == rhs.min() && max() == rhs.max(); }

    bool operator != (const bbox2& rhs) const { return  !(*this == rhs);  }

protected:
    POINT min_;
    POINT max_;
};
