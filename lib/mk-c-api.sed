#!/bin/sh

s|PRAGMA \(.*\)|#\1|g
s|COMMENT\(.*\)|//\1|g
s/EMPTYLINE//g
s/^XSS_DLL_IMPORT/    XSS_DLL_IMPORT/g
s/^inline/    inline/g
