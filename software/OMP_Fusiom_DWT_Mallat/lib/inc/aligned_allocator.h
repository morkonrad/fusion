#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <cstdlib>
#include <new>
#include <limits>
// source code used from http://jmabille.github.io/blog/2014/12/06/aligned-memory-allocator/
// and modified a little bit

template <class T, int N>
    class aligned_allocator
    {

    public:

        typedef T value_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        template <class U>
            struct rebind
            {
                using other =  aligned_allocator<U,N>;
            };

        inline aligned_allocator() throw() {}
    //   inline aligned_allocator(const aligned_allocator&) throw() {}


        template <class U> aligned_allocator(const aligned_allocator<U,N>&) throw() {}

    //     ~aligned_allocator() throw() {}

  //      inline pointer address(reference r) { return &r; }
  //      inline const_pointer address(const_reference r) const { return &r; }

        inline pointer allocate(size_type n, typename std::allocator<void>::const_pointer hint = 0);
        inline void deallocate(pointer p, size_type);

       // inline void construct(pointer p, const_reference value) { new (p) value_type(value); }
       // inline void destroy(pointer p) { p->~value_type(); }

        inline size_type max_size() const throw() { return std::numeric_limits<size_t>::max() / sizeof(T); }

        inline bool operator==(const aligned_allocator&) { return true; }
        inline bool operator!=(const aligned_allocator& rhs) { return !operator==(rhs); }
    };


template <class T, int N>
typename aligned_allocator<T,N>::pointer aligned_allocator<T,N>::allocate(size_type n, typename std::allocator<void>::const_pointer/* hint*/)
{
        void* res = 0;
        void* ptr = std::malloc(sizeof(T)*n+N);
        if(ptr != 0)
        {
            res = reinterpret_cast<void*>((reinterpret_cast<size_t>(ptr) & ~(size_t(N-1))) + N);
            *(reinterpret_cast<void**>(res) - 1) = ptr;
        }
        else
        {
          // throw std::bad_alloc();
        }
        return reinterpret_cast<pointer>(res);
}

template <class T, int N>
void aligned_allocator<T,N>::deallocate(pointer p, size_type)
{
   if(p != 0)
   {
      std::free(*(reinterpret_cast<void**>(p)-1));
   }
}

#endif // ALIGNED_ALLOCATOR_H
