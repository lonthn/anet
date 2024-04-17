#include "include/util/iobyte.h"

#include <iostream>
#include <cassert>
#include <cstring>

#define self (*this)

obstream::obstream(obstream &&stream) {
    _buffer = stream._buffer;
    stream._buffer = nullptr;

    _size = stream._size;
    stream._size = 0;

    _length = stream._length;
    stream._length = 0;
}

obstream::obstream(size_t capacity) {
    if (capacity == 0 || capacity > static_cast<size_t>(-1)) {
        return;
    }

    _buffer = (unsigned char *)malloc(capacity);
    _size = capacity;
}

obstream::~obstream() {
    if (_buffer != nullptr) {
        free(_buffer);
    }
}

void obstream::initial(size_t capacity) {
    if (_buffer == nullptr) {
        resize(capacity);
    }
}

void obstream::write(void* data, size_t size) {
    if (size == 0) {
        return;
    }

    if (_length+size > _size) {
        if (resize(_size + size)) {
            return;
        }
    }

    memcpy(_buffer + _length, data, size);
    _length += size;
}

obstream& obstream::operator << (char val) {
    write(&val, 1);
    return self;
}
obstream& obstream::operator << (bool val) {
    char yes = 1, no = 0;
    write(&(val ? yes : no), 1);
    return self;
}
obstream& obstream::operator << (int16_t val) {
    write(&val, 2);
    return self;
}
obstream& obstream::operator << (uint16_t val) {
    write(&val, 2);
    return self;
}
obstream& obstream::operator << (int32_t val) {
    write(&val, 4);
    return self;
}
obstream& obstream::operator << (uint32_t val) {
    write(&val, 4);
    return self;
}
obstream& obstream::operator << (int64_t val) {
    auto temp = /*htonll*/((uint64_t)val);
    write(&temp, 8);
    return self;
}
obstream& obstream::operator << (uint64_t val) {
    auto temp = /*htonll*/(val);
    write(&temp, 8);
    return self;
}
obstream& obstream::operator << (float val) {
    write(&val, 4);
    return self;
}
obstream& obstream::operator << (double val) {
    write(&val, 8);
    return *this;
}
obstream& obstream::operator<<(std::string& val) {
#ifdef _WIN32
    std::string tempVal = val; //strutil::gbkToUtf8(val.c_str());
#else
    std::string& tempVal = val;
#endif

    uint32_t len = (uint32_t)tempVal.length();
    write(&len, 4);

    if (len != 0) {
        write((void *)tempVal.data(), len);
    }

    return self;
}
obstream& obstream::operator << (const std::string& val) {
#ifdef _WIN32
    std::string tempVal = val; //strutil::gbkToUtf8(val.c_str());
#else
    const std::string& tempVal = val;
#endif

    uint32_t len = (uint32_t)tempVal.length();
    write(&len, 4);

    if (len != 0) {
        write((void *)tempVal.c_str(), len);
    }

    return *this;
}

//obstream& obstream::operator << (const std::string_view val) {
//    std::string copy(val);
//    self << copy;
//    return self;
//}

bool obstream::resize(size_t new_size) {
    size_t old_size = _size;
    if (new_size <= old_size) {
        return false;
    }
    if (old_size == 0) {
        old_size = new_size;
    }
    size_t alloc_size = old_size;
    while (new_size > alloc_size) {
        alloc_size *= 2;
    }

    if (alloc_size > static_cast<size_t>(-1)) {
        alloc_size = static_cast<size_t>(-1);
        if (alloc_size < new_size) {
            return true;
        }
    }

    auto buffer = (unsigned char *)realloc(_buffer, alloc_size);
    _buffer = buffer;
    _size = alloc_size;
    return false;
}

/* End class byte_writer implementation */


/* Class byte_reader implementation */
ibstream::ibstream(std::string &data) :
        ibstream((void*)data.data(), (unsigned int) data.length()) {
}

//ibstream::ibstream(std::string_view &data) :
//        ibstream((void *) data.data(), (unsigned int) data.length()) {
//}

ibstream::ibstream(void *data, size_t length) :
        _begin((unsigned char *) data),
        _end(_begin + length),
        _pointer(_begin) {
}

void ibstream::seek(int offset, seek_origin origin) {
    if (is_data_lack(self)) {
        return;
    }

    if (origin == seek_origin::beg) {
        _pointer = _begin + offset;
    } else if (origin == seek_origin::end) {
        _pointer = _end + offset;
    } else {
        _pointer += offset;
    }

    if (_pointer < _begin) {
        std::cerr << "-> Seek] Move pointer out range, pointer: "
                    << (int32_t)(_pointer-_begin)
                    << ", length: " << (int32_t) (_end-_begin)<<"." << std::endl;
    } else if (_pointer > _end) {
        _pointer = _end + 1;

        std::cerr << "-> Seek] Move pointer out range, pointer: "
                    << (int32_t)(_pointer-_begin)
                    << ", length: " << (int32_t) (_end-_begin)<<"." << std::endl;
    }
}

void ibstream::read(void *data, size_t length) {
    if (!is_data_lack(self)) {
        if ((_pointer+length) > _end) {
            _pointer = _end + 1;
            std::cerr << "-> Read] Read pointer out range, pointer: "
                      << (int32_t)(_pointer-_begin)
                      << ", length: " << (int32_t)(_end-_begin) << "." << std::endl;
        } else {
            memcpy(data, _pointer, length);
            _pointer += length;
        }
    }
}

ibstream& ibstream::operator >> (char &val) {
    read(&val, 1);
    return self;
}
ibstream& ibstream::operator >> (bool &val) {
    char temp = 0;
    read(&temp, 1);
    val = temp == 1;
    return self;
}
ibstream& ibstream::operator >> (short &val) {
    read(&val, 2);
    //val = ntohs(val);
    return self;
}
ibstream& ibstream::operator >> (unsigned short &val) {
    read(&val, 2);
    //val = ntohs(val);
    return self;
}
ibstream& ibstream::operator >> (int &val) {
    read(&val, 4);
    //val = (int) ntohl(val);
    return self;
}
ibstream& ibstream::operator >> (unsigned int &val) {
    read(&val, 4);
    //val = ntohl(val);
    return self;
}
ibstream& ibstream::operator >> (int64_t &val) {
    read(&val, 8);
    val = (long long) /*ntohll*/(val);
    return self;
}
ibstream& ibstream::operator >> (uint64_t &val) {
    read(&val, 8);
    val = /*ntohll*/(val);
    return self;
}
ibstream& ibstream::operator >> (float &val) {
    read(&val, 4);
    return self;
}
ibstream& ibstream::operator >> (double &val) {
    read(&val, 8);
    return self;
}
ibstream& ibstream::operator >> (std::string &val) {
    uint32_t length;
    if (is_data_lack(self >> length) || length == 0) {
        return self;
    }

    if ((_pointer+length) > _end) {
        _pointer = _end + 1;
        std::cerr << "'>> string' is data lack." << std::endl;
        return self;
    }

    val.assign((const char*)_pointer, length);
#ifdef _WIN32
	  //val = strutil::utf8ToGBK(val.c_str());
#endif
    _pointer += length;
    return self;
}

int8_t ibstream::read_int8() {
    int8_t res;
    read(&res, 1);
    return res;
}

int32_t ibstream::read_int32() {
    int32_t res;
    read(&res, 4);
    return res;
}

uint32_t ibstream::read_uint32() {
    uint32_t res;
    read(&res, 4);
    return res;
}

int64_t ibstream::read_int64() {
    uint64_t ret;
    read(&ret, 8);
    return (int64_t) /*ntohll*/(ret);
}
uint64_t ibstream::read_uint64() {
    uint64_t ret;
    read(&ret, 8);
    return /*ntohll*/(ret);
}
float ibstream::read_float32() {
    float ret;
    read(&ret, 4);
    return ret;
}
double ibstream::read_float64() {
    double ret;
    read(&ret, 8);
    return ret;
}
bool ibstream::read_bool() {
    char temp = 0;
    read(&temp, 1);
    return (bool) temp;
}
std::string ibstream::read_string() {
    uint32_t length;
    if (is_data_lack(self >> length) || length == 0) {
        return {};
    }

    if ((_pointer+length) > _end) {
        _pointer = _end + 1;
        assert(false);
        return {};
    }

    std::string ret((const char *) _pointer, length);
#ifdef _WIN32
    //ret = strutil::utf8ToGBK(ret.c_str());
#endif
    _pointer += length;
    return ret;
}

/* End class byte_reader implementation */

#undef self

int bytes_len(const std::string& str) {
    //return 4 + (int) str.length();

    int len = (int) str.length();
    if (len == 0) {
        return len;
    } else if (len < 126) {
        return 1 + len;
    } else if (len <= UINT16_MAX) {
        return 3 + len;
    } else {
        return 5 + len;
    }
}