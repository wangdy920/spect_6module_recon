# Convert a binary file into a C++ array.
#   cmake -DINPUT=... -DOUTPUT=... -DSYMBOL=... -P EmbedBin.cmake

if(NOT INPUT OR NOT OUTPUT OR NOT SYMBOL)
  message(FATAL_ERROR "EmbedBin.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()
if(NOT EXISTS "${INPUT}")
  message(FATAL_ERROR "EmbedBin input missing: ${INPUT}")
endif()

file(READ "${INPUT}" _hex HEX)
string(LENGTH "${_hex}" _nhex)
math(EXPR _nbytes "${_nhex} / 2")
string(REGEX REPLACE "([0-9A-Fa-f][0-9A-Fa-f])" "0x\\1," _body "${_hex}")
string(REGEX REPLACE "((0x[0-9A-Fa-f][0-9A-Fa-f],){16})" "\\1\n" _body "${_body}")

file(WRITE "${OUTPUT}"
"#include <cstddef>\n\n"
"extern const unsigned char ${SYMBOL}[] = {\n${_body}\n};\n"
"extern const unsigned long long ${SYMBOL}_size = ${_nbytes}ull;\n"
)
