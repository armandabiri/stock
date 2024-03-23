cmake_minimum_required(VERSION 3.12)

function(has_files_in_folder folder_path result_var)
  # Get a list of files in the directory and its subdirectories
  file(GLOB_RECURSE files "${folder_path}/*")

  # Check if the list is not empty
  if(files)
    set(${result_var} TRUE PARENT_SCOPE)
  else()
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(print_header MESSAGE)
  string(LENGTH "${MESSAGE}" MESSAGE_LENGTH)
  set(N 40)

  math(EXPR PADDING_LENGTH1 "(2*${N})")
  math(EXPR PADDING_LENGTH "( ${N} - ${MESSAGE_LENGTH}/2 )")

  string(REPEAT "_" ${PADDING_LENGTH1} PADDING1)
  string(REPEAT " " ${PADDING_LENGTH} PADDING)
  message(STATUS "${PADDING1}")
  message(STATUS "${PADDING}${MESSAGE}${PADDING}")
  message(STATUS "${PADDING1}")
endfunction()

function(get_generators)
  set(GENERATORS_LIST)
  execute_process(
    COMMAND cmake --help
    COMMAND findstr "Generators"
    OUTPUT_VARIABLE CMAKE_HELP_OUTPUT
    RESULT_VARIABLE CLONE_RESULT
    ERROR_QUIET
  )

  if(CLONE_RESULT EQUAL 0)
    string(REGEX MATCHALL ".*Generator for .*" GENERATORS_LIST ${CMAKE_HELP_OUTPUT})

    foreach(GENERATOR ${GENERATORS_LIST})
      string(REGEX REPLACE ".*Generator for (.*):.*" "\\1" GENERATOR ${GENERATOR})
      list(APPEND AVAILABLE_GENERATORS ${GENERATOR})
    endforeach()

    message("Available Generators: ${AVAILABLE_GENERATORS}")
  else()
    message("Failed to retrieve generators.")
  endif()

  set(GENERATORS_LIST ${GENERATORS_LIST} PARENT_SCOPE)
endfunction()

function(parse_line line PATTERN result found)
  string(REGEX MATCH "${PATTERN}" MATCH "${line}")

  set(${found} FALSE PARENT_SCOPE)

  if(MATCH)
    set(${found} TRUE PARENT_SCOPE)

    if(NOT ${CMAKE_MATCH_1} STREQUAL "")
      set(${result} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
      set(${result} "N/A" PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(read_requirements FILENAME)
  file(READ ${FILENAME} FILE_CONTENT)
  string(REGEX REPLACE "\r?\n" ";" LINES ${FILE_CONTENT})

  # Variables to store the parsed data
  set(REPOS "GIT" "BRANCH" "FORCE_UPDATE" "CMAKE_CONFIG" "BUILD_CONFIG" "BUILD_TYPE")

  set(GENERAL_VALUES "GENERATOR" "ARCH" "TOOLSET")

  set(GENERATOR "Visual Studio 15 2017")
  set(ARCH "x64")
  set(TOOLSET "v143")

  foreach(repo ${REPOS})
    set(${repo}_LIST)
  endforeach()

  # Loop through each line
  foreach(line ${LINES})
    # Check if the line contains any information
    if(NOT line STREQUAL "")
      set(FOUND FALSE)

      foreach(gen ${GENERAL_VALUES})
        parse_line("${line}" "^${gen}: *(.*)" VALUE FOUND)

        if(FOUND)
          set(${gen} ${VALUE})
          break()
        endif()
      endforeach()

      if(FOUND)
        continue()
      endif()

      foreach(repo ${REPOS})
        parse_line("${line}" "^${repo}: *(.*)" VALUE FOUND)

        if(FOUND)
          list(APPEND ${repo}_LIST ${VALUE})
          set(${LIST_NAME} ${${LIST_NAME}} PARENT_SCOPE)
          break()
        endif()
      endforeach()

      if(FOUND)
        continue()
      endif()
    endif()
  endforeach()

  foreach(repo ${REPOS})
    set(${repo}_LIST ${${repo}_LIST} PARENT_SCOPE)
  endforeach()

  foreach(gen ${GENERAL_VALUES})
    set(${gen} ${${gen}} PARENT_SCOPE)
  endforeach()
endfunction()

function(clone_git GIT BUILD_DIR REPO_NAME FORCE_UPDATE BRANCH)
  # Determine the clone command based on the presence of the branch and update flag
  if(NOT EXISTS "${BUILD_DIR}/${REPO_NAME}")
    set(CLONE_COMMAND git clone)

    # Add branch if defined
    if(NOT(BRANCH STREQUAL "N/A"))
      message("Cloning the branch [${BRANCH}] of [${REPO_NAME}] from [${GIT}] in the directory [${BUILD_DIR}/${REPO_NAME}]")
      list(APPEND CLONE_COMMAND "--branch" ${BRANCH})
    else()
      message("Cloning the main branch of [${REPO_NAME}] from [${GIT}] in the directory [${BUILD_DIR}/${REPO_NAME}]")
    endif()

    # Append GIT and BUILD_DIR to the clone command
    list(APPEND CLONE_COMMAND ${GIT} ${BUILD_DIR}/${REPO_NAME})
  else()
    if(FORCE_UPDATE EQUAL "ON")
      message("Updating the branch [${BRANCH}] of [${REPO_NAME}] from [${GIT}] in the directory [${BUILD_DIR}/${REPO_NAME}]")
      set(CLONE_COMMAND git --git-dir=${BUILD_DIR}/${REPO_NAME}/.git pull)
    else()
      message("Skipping [${REPO_NAME}]...")
      return()
    endif()
  endif()

  # Execute the clone command
  execute_process(
    COMMAND ${CLONE_COMMAND}
    RESULT_VARIABLE CLONE_RESULT
    ERROR_QUIET
  )

  # Check for errors during cloning
  if(NOT ${CLONE_RESULT} EQUAL "0")
    message(FATAL_ERROR "Error: Failed to clone/update ${GIT} using command: ${CLONE_COMMAND}")
  endif()
endfunction()

function(cmake_build BUILD_DIR REPO_NAME FORCE_UPDATE CMAKE_CONFIG BUILD_CONFIG BUILD_TYPE GENERATOR ARCH CMAKE_PREFIX_PATH)
  # Determine the build directory path
  set(PATH "${BUILD_DIR}/${REPO_NAME}/build")

  # Initialize variables for messaging and build command
  set(MSG "Creating")
  set(BUILD_COMMAND "")

  # Check if the build directory exists and if update is requested
  if(EXISTS ${PATH})
    if(FORCE_UPDATE STREQUAL "ALL")
      set(MSG "Removing")
      set(folderToDelete ${PATH})

      while(EXISTS ${folderToDelete})
        # Delete the folder
        message("Deleting folder: ${folderToDelete}")
        file(REMOVE_RECURSE ${folderToDelete})

        # Sleep for a short duration before checking again
        # This helps to reduce CPU usage during the loop
        # Adjust the sleep duration according to your requirements
        # For example, sleep for 1 second
        execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 1)
      endwhile()
    endif()
  else()
    # Create the build directory if it doesn't exist
    file(MAKE_DIRECTORY ${PATH})
  endif()

  while(NOT EXISTS ${PATH})
    file(MAKE_DIRECTORY ${PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 1)
  endwhile()

  build_and_install_project(${PATH} ${CMAKE_CONFIG} ${BUILD_CONFIG} ${BUILD_DIR} ${REPO_NAME} ${BUILD_TYPE} ${FORCE_UPDATE} ${GENERATOR} ${ARCH} ${TOOLSET} ${CMAKE_PREFIX_PATH})
endfunction()

function(build_and_install_project PATH CMAKE_CONFIG BUILD_CONFIG BUILD_DIR REPO_NAME BUILD_TYPE FORCE_UPDATE GENERATOR ARCH TOOLSET CMAKE_PREFIX_PATH)
  # Execute the build command
  if(CMAKE_CONFIG STREQUAL "N/A")
    set(CMAKE_CONFIG "")
  endif()

  if(BUILD_TYPE STREQUAL "N/A")
    set(BUILD_TYPE "debug")
  endif()

  has_files_in_folder("${BUILD_DIR}/${REPO_NAME}/build" has_files)

  if(${FORCE_UPDATE} STREQUAL "REBUILD" OR ${FORCE_UPDATE} STREQUAL "ALL" OR NOT has_files)
    set(command cmake .. -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}/${REPO_NAME}/build/install -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -G ${GENERATOR} -A ${ARCH} -T ${TOOLSET} -DCMAKE_PREFIX_PATH=${DCMAKE_PREFIX_PATH} ${CMAKE_CONFIG})
    message(">>>>>>>>>>>>>>>>>>>>>> Configuring [${REPO_NAME}] >>>>>>>>>>>>>>>>>>> ${command}")

    execute_process(
      COMMAND ${command}
      WORKING_DIRECTORY ${PATH}
      RESULT_VARIABLE CONFIGURE_RESULT
    )

    if(${CONFIGURE_RESULT} EQUAL 0)
      if(BUILD_CONFIG STREQUAL "N/A")
        set(BUILD_CONFIG "")
      endif()

      foreach(config IN ITEMS debug release debug)
        set(command cmake --build . --parallel 24 --config ${config} ${BUILD_CONFIG})
        message(">>>>>>>>>>>>>>>>>>>>>> Building [${REPO_NAME} :: ${config} mode] >>>>>>>>>>>>>>>>>>> ${command}")

        execute_process(
          COMMAND ${command}
          WORKING_DIRECTORY ${PATH}
          RESULT_VARIABLE BUILD_RESULT
        )

        if(${BUILD_RESULT} EQUAL 0)
        else()
          message(FATAL_ERROR "Error: Failed to build [${REPO_NAME}].")
        endif()
      endforeach()
    else()
      message(FATAL_ERROR "Error: Failed to config [${REPO_NAME}].")
    endif()
  endif()

  if((${FORCE_UPDATE} STREQUAL "REBUILD") OR ${FORCE_UPDATE} STREQUAL "INSTALL" OR ${FORCE_UPDATE} STREQUAL "ALL" OR ${FORCE_UPDATE} STREQUAL "ON")
    # foreach(BUILD_TYPE IN ITEMS debug release)
    set(command cmake --build . --target install --config ${BUILD_TYPE})
    message(">>>>>>>>>>>>>>>>>>>>>> Installing [${REPO_NAME} :: ${BUILD_TYPE} mode] >>>>>>>>>>>>>>>>>>> ${command}")

    execute_process(
      COMMAND ${command}
      WORKING_DIRECTORY ${PATH}
      RESULT_VARIABLE INSTALL_RESULT
    )

    if(NOT ${INSTALL_RESULT} EQUAL 0)
      message(FATAL_ERROR "Error: Failed to install [${REPO_NAME}]")
    endif()

    # endforeach()
  endif()
endfunction()

function(resolve_dependencies FILE_PATH BUILD_DIR)
  print_header("Build Dependencies from [${FILE_PATH}]")

  set(REPO_NAME_LIST)
  read_requirements(${FILE_PATH})

  list(LENGTH GIT_LIST URL_COUNT)
  math(EXPR URL_COUNT "${URL_COUNT} - 1")

  foreach(index RANGE ${URL_COUNT})
    list(GET GIT_LIST ${index} GIT)
    list(GET BRANCH_LIST ${index} BRANCH)
    list(GET FORCE_UPDATE_LIST ${index} FORCE_UPDATE)
    list(GET BUILD_CONFIG_LIST ${index} BUILD_CONFIG)
    list(GET BUILD_TYPE_LIST ${index} BUILD_TYPE)
    list(GET CMAKE_CONFIG_LIST ${index} CMAKE_CONFIG)

    # Perform regex match
    string(REGEX MATCH "(^https:\/\/.*\/(.*).git)(.*)?$" MATCHES ${GIT})
    set(GIT ${CMAKE_MATCH_1}) # Group 2: branch name
    set(REPO_NAME ${CMAKE_MATCH_2}) # Group 2: branch name

    list(APPEND REPO_NAME_LIST "${REPO_NAME}")

    message(">>>>> (${index}/${URL_COUNT}) Processing [${GIT}] with -B:[${BUILD_DIR}] -N:[${REPO_NAME}] -U:[${FORCE_UPDATE}] \n CMAKE_CONFIG:[${CMAKE_CONFIG}] \n BUILD_CONFIG:[${BUILD_CONFIG}]\n BUILD_TYPE:[${BUILD_TYPE}]")

    clone_git(${GIT} ${BUILD_DIR} ${REPO_NAME} ${FORCE_UPDATE} ${BRANCH})
    cmake_build(${BUILD_DIR} ${REPO_NAME} ${FORCE_UPDATE} ${CMAKE_CONFIG} ${BUILD_CONFIG} ${BUILD_TYPE} ${GENERATOR} ${ARCH} ${TOOLSET} ${CMAKE_PREFIX_PATH})

    list(APPEND CMAKE_PREFIX_PATH "${BUILD_DIR}/${REPO_NAME}/build/install")
  endforeach()

  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
  print_header("Finished building dependencies!!")
endfunction()
