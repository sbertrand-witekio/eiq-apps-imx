#!/bin/sh -ex
#
# SPDX-License-Identifier: LGPL-2.0+
# Copyright 2021 NXP

# setup i.MX Linux BSP toolchain
#source /opt/fsl-imx-xwayland/5.4-zeus/environment-setup-aarch64-poky-linux

# for m4
mkdir -p m4

# aclocal
ACLOCAL="aclocal --system-acdir=${SDKTARGETSYSROOT}/usr/share/aclocal/"

# aclocal path has 'automake --version'
AUTOMAKE_VERSION=`automake --version | sed -e '1{s/.* //;s/\.[0-9]\+$//};q'`
if [ -d ${OECORE_NATIVE_SYSROOT}/usr/share/aclocal-${AUTOMAKE_VERSION} ]; then
	ACLOCAL="$ACLOCAL --automake-acdir=${OECORE_NATIVE_SYSROOT}/usr/share/aclocal-${AUTOMAKE_VERSION}"
fi

# autoreconf option
ACPATHS="-I ${OECORE_NATIVE_SYSROOT}/usr/share/aclocal/"

# autoreconf
ACLOCAL="$ACLOCAL" autoreconf -Wcross --verbose --install --force --exclude=autopoint ${ACPATHS}

# configure
#./configure --host=aarch64-poky-linux
