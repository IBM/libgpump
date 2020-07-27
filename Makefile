# Copyright (C) IBM Corporation 2018. All Rights Reserved
#
#    This program is licensed under the terms of the Eclipse Public License
#    v1.0 as published by the Eclipse Foundation and available at
#    http://www.eclipse.org/legal/epl-v10.html
#
#    
#    
# $COPYRIGHT$
all:
	echo "Makefile is DEAD, use Makefile.build instead - gpaulsen 8/18/2019"
	exit
	export INCDIR=${BASE}/dependencies/include/cuda && \
	  make -C src all
	echo make -C test all
	echo make -C examples all
	echo make -C mod all
	
clean:
	echo "Makefile is DEAD, use Makefile.build instead - gpaulsen 8/18/2019"
	exit
	make -C src clean
	echo make -C test clean
	echo make -C examples clean
	echo make -C mod clean
