// Exception.cpp

#include "omnilearn/Exception.hh"



omnilearn::Exception::Exception(std::string const& msg):
_msg("[Brain.Exception : " + msg + "]")
{
}


const char* omnilearn::Exception::what() const noexcept
{
    return _msg.c_str();
}