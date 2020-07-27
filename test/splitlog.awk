#!/bin/awk -f
# Copyright (C) IBM Corporation 2018. All Rights Reserved
#
#    This program is licensed under the terms of the Eclipse Public License
#    v1.0 as published by the Eclipse Foundation and available at
#    http://www.eclipse.org/legal/epl-v10.html
#
#    
#    
# $COPYRIGHT$
{
  one=$1
  if ( substr(one,1,1) == "[" )
  {
    tag=substr(one,2,length(one)-2)
    print $0 >"part." tag
  }
}
