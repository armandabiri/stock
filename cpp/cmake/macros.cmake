cmake_minimum_required(VERSION 3.12)

function(add_subdirectories)
  set(curdir ${CMAKE_CURRENT_SOURCE_DIR})
  file(GLOB children LIST_DIRECTORIES true RELATIVE ${curdir} ${curdir}/*)

  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      if(NOT(${child} MATCHES "^\\." OR ${child} MATCHES "^_"))
        message(">>>> Explore subdirectory [${child}]")
        add_subdirectory(${child})
      endif()
    endif()
  endforeach()
endfunction()

function(list_subdirectories result curdir)
  file(GLOB children LIST_DIRECTORIES true RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")

  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()

  set(${result} ${dirlist} PARENT_SCOPE)
endfunction()
