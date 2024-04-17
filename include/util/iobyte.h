//
// Created by zeqi.luo on 2020/8/5.
//

#ifndef IO_BYTE_STREAM
#define IO_BYTE_STREAM

#include <string>
#include <memory>

enum seek_origin {
  beg, end, cur
};

class ibstream {
public:
  ibstream() = delete;
  explicit ibstream(std::string& data);
  /*explicit ibstream(std::string_view& data);*/
  ibstream(void* buffer, size_t length);

  static bool is_data_lack(const ibstream& in) {
    return in._pointer > in._end;
  }

public:
  void seek(int offset, seek_origin origin);
  void read(void* data, size_t length);

  ibstream& operator>>(char& val);
  ibstream& operator>>(bool& val);
  ibstream& operator>>(int16_t& val);
  ibstream& operator>>(uint16_t& val);
  ibstream& operator>>(int32_t& val);
  ibstream& operator>>(uint32_t& val);
  ibstream& operator>>(int64_t& val);
  ibstream& operator>>(uint64_t& val);
  ibstream& operator>>(float& val);
  ibstream& operator>>(double& val);
  ibstream& operator>>(std::string& val);

  int8_t read_int8();
  int32_t read_int32();
  uint32_t read_uint32();
  int64_t read_int64();
  uint64_t read_uint64();
  float read_float32();
  double read_float64();
  bool read_bool();

  std::string read_string();

private:
  unsigned char* _begin = nullptr;
  unsigned char* _end = nullptr;
  unsigned char* _pointer = nullptr;
};


class obstream : public std::allocator<unsigned char> {
public:
  obstream() = default;
  obstream(obstream&& stream);
  obstream(size_t capacity);
  ~obstream();

  void initial(size_t capacity);

  std::pair<char*, int> data() {
    return std::make_pair((char*) _buffer, _length);
  }
  unsigned char* bytes() {
    return _buffer;
  }
  size_t length() {
    return _length;
  }

public:
  void write(void* data, size_t size);

  obstream& operator<<(char val);
  obstream& operator<<(bool val);
  obstream& operator<<(int16_t val);
  obstream& operator<<(uint16_t val);
  obstream& operator<<(int32_t val);
  obstream& operator<<(uint32_t val);
  obstream& operator<<(int64_t val);
  obstream& operator<<(uint64_t val);
  obstream& operator<<(float val);
  obstream& operator<<(double val);
  obstream& operator<<(std::string& val);
  obstream& operator<<(const std::string& val);
//  obstream& operator<<(std::string_view val);

  bool resize(size_t new_size);

private:
  unsigned char* _buffer = nullptr;
  size_t _length = 0;
  size_t _size = 0;
};

int bytes_len(const std::string&);

#endif // #ifndef IO_BYTE_STREAM