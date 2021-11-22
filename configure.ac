# SPDX-License-Identifier: LGPL-2.0+
# Copyright 2021 NXP

AC_PREREQ(2.59)

dnl initialize autoconf
AC_INIT(i.MX GStreamer NN Inference demo, 0.9.1, http://www.nxp.com, gstnninferencedemo)

dnl initialize automake
AM_INIT_AUTOMAKE([foreign tar-ustar])

dnl disable-static
LT_INIT([disable-static])

dnl our libraries and install dirs use GST_API_VERSION in the filename
dnl to allow side-by-side installation of different API versions
GST_API_VERSION=1.0
AC_SUBST(GST_API_VERSION)
AC_DEFINE_UNQUOTED(GST_API_VERSION, "$GST_API_VERSION", [GStreamer API Version])

dnl gstreamer version requirement
GST_REQ=1.6.0
PKG_CHECK_MODULES(GST, gstreamer-$GST_API_VERSION >= $GST_REQ,
  [HAVE_GST_CHECK="yes"], [HAVE_GST_CHECK="no"
    AC_MSG_NOTICE([no gstreamer-$GST_API_VERSION >= $GST_REQ (GStreamer) found])
  ])
AM_CONDITIONAL(HAVE_GST_CHECK, test "x$HAVE_GST_CHECK" = "xyes")

dnl m4 settings
AC_CONFIG_MACRO_DIR([m4])
AC_SUBST(ACLOCAL_AMFLAGS, "-I m4")

dnl check enable feature
AC_DEFUN([CHECK_ENABLE_FEATURE],
[
  AC_ARG_ENABLE(
    $1,
    [AS_HELP_STRING([--enable-[$1]], [$2])],
    [enable_[$1]=$enableval],
    [enable_[$1]=no])
  if test "$enable_[$1]" = "yes"; then
    for h in "$4"; do
      AC_CHECK_HEADERS([$h], HAVE_[$3]=yes && break, HAVE_[$3]=no)
    done
  fi
  AM_CONDITIONAL(ENABLE_[$3], test "x$HAVE_[$3]" = "xyes")
])

dnl AM_MAINTAINER_MODE provides the option to enable maintainer mode
AM_MAINTAINER_MODE

dnl check for tools
AC_PROG_CC
AC_PROG_CXX
AM_PROG_CC_C_O
AC_PROG_LIBTOOL

dnl Check for pkgconfig first
AC_CHECK_PROG(HAVE_PKGCONFIG, pkg-config, yes, no)

dnl Give error and exit if we don't have pkgconfig
if test "x$HAVE_PKGCONFIG" = "xno"; then
  AC_MSG_ERROR(you need to have pkgconfig installed !)
fi

dnl Now we're ready to ask for gstreamer libs and cflags
dnl And we can also ask for the right version of gstreamer
GST_REQUIRED=1.0
GST_MAJORMINOR=1.0
PKG_CHECK_MODULES(GST, gstreamer-$GST_MAJORMINOR >= $GST_REQUIRED, HAVE_GST=yes, HAVE_GST=no)

dnl Give error and exit if we don't have gstreamer
if test "x$HAVE_GST" = "xno"; then
  AC_MSG_ERROR(you need gstreamer development packages installed !)
fi

dnl make GST_CFLAGS and GST_LIBS available
AC_SUBST(GST_CFLAGS)
AC_SUBST(GST_LIBS)

dnl make GST_MAJORMINOR available in Makefile.am
AC_SUBST(GST_MAJORMINOR)

dnl If we need them, we can also use the plugin libraries
PKG_CHECK_MODULES(GST_BASE, gstreamer-base-$GST_MAJORMINOR >= $GST_REQUIRED, HAVE_GST_BASE=yes, HAVE_GST_BASE=no)

dnl Give a warning if we don't have gstreamer libs
if test "x$HAVE_GST_BASE" = "xno"; then
  AC_MSG_ERROR(no GStreamer base found)
fi

dnl make GST_BASE_CFLAGS and GST_BASE_LIBS available
AC_SUBST(GST_BASE_CFLAGS)
AC_SUBST(GST_BASE_LIBS)

dnl If we need them, we can also use the plugin libraries
PKG_CHECK_MODULES(GST_PLUGINS_BASE, gstreamer-plugins-base-$GST_MAJORMINOR >= $GST_REQUIRED, HAVE_GST_PLUGINS_BASE=yes, HAVE_GST_PLUGINS_BASE=no)

dnl Give a warning if we don't have gstreamer libs
if test "x$HAVE_GST_PLUGINS_BASE" = "xno"; then
  AC_MSG_ERROR(no GStreamer plugin base libs found)
fi

dnl make GST_PLUGINS_BASE_CFLAGS and GST_PLUGINS_BASE_LIBS available
AC_SUBST(GST_PLUGINS_BASE_CFLAGS)
AC_SUBST(GST_PLUGINS_BASE_LIBS)

dnl add include path for imx
CFLAGS="$CFLAGS -I$SDKTARGETSYSROOT/usr/include/imx"

dnl check ion allocator header files
old_CFLAGS=$CFLAGS
PKG_CHECK_MODULES(GST, gstreamer-$GST_MAJORMINOR >= 1.14, CFLAGS="$CFLAGS `$PKG_CONFIG --cflags gstreamer-allocators-1.0`", CFLAGS="$CFLAGS `$PKG_CONFIG --cflags gstreamer-bad-allocators-1.0`")
AC_CHECK_HEADERS([linux/ion.h gst/allocators/gstionmemory.h], HAVE_ION="yes", HAVE_ION="no")
AM_CONDITIONAL(USE_ION, test "x$HAVE_ION" = "xyes")
CFLAGS=$old_CFLAGS

dnl set the plugindir where plugins should be installed
plugindir="\$(libdir)/gstreamer-$GST_MAJORMINOR"
AC_SUBST(plugindir)

dnl set proper LDFLAGS for plugins
GST_PLUGIN_LDFLAGS='-module -avoid-version -export-symbols-regex gst_plugin_.*'
AC_SUBST(GST_PLUGIN_LDFLAGS)

AC_OUTPUT(
Makefile
src/Makefile)
