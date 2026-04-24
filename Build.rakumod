#| Build.rakumod for ONNX::Native.
#|
#| Two paths, tried in order:
#|
#|   1. Prebuilt binary download from Microsoft's official ONNX
#|      Runtime GitHub Releases. One tarball per platform contains
#|      libonnxruntime + its headers; we stage both, then compile
#|      src/onnx_native_shim.c against them and place the resulting
#|      libonnx_native_shim.{dylib,so,dll} next to libonnxruntime
#|      so it resolves via @loader_path / $ORIGIN / sibling-DLL.
#|      SHA256 verified against resources/checksums.txt.
#|
#|   2. Fallback: look for a system libonnxruntime install
#|      (Homebrew / apt / manual) and compile the shim against it.
#|      This keeps ONNX::Native installable on systems that already
#|      have ORT available but aren't on one of the platforms we
#|      publish prebuilts for (older Linux glibc / niche archs).
#|
#| Linux prebuilts from Microsoft are built on ubuntu-20.04-ish
#| (glibc 2.31+). On very old systems the prebuilt libonnxruntime
#| fails to load with "GLIBC_2.xx not found"; we detect via
#| `ldd --version` and short-circuit to the system-ORT fallback.
#|
#| Why we don't use META6 resources for the libs: zef hashes every
#| staged resource filename, which would break both libonnxruntime's
#| own versioned symlink (libonnxruntime.dylib → libonnxruntime.1.20.0.dylib)
#| and the shim's @loader_path/$ORIGIN reference to it. Same
#| rationale as Vips-Native's Build.rakumod.
#|
#| Env-var knobs:
#|
#|   ONNX_NATIVE_PREFER_SYSTEM=1   skip prebuilt download, compile
#|                                 shim against system libonnxruntime
#|   ONNX_NATIVE_BINARY_ONLY=1     refuse to fall back to system
#|                                 libonnxruntime; fail if prebuilt
#|                                 unavailable
#|   ONNX_NATIVE_BINARY_URL=<url>  override GH release base URL
#|                                 (defaults to the microsoft/
#|                                 onnxruntime upstream releases)
#|   ONNX_NATIVE_CACHE_DIR=<path>  override download cache dir
#|   ONNX_NATIVE_DATA_DIR=<path>   override staged-libs base dir
#|                                 (defaults to XDG_DATA_HOME)
#|   ONNX_NATIVE_LIB_DIR=<path>    (runtime) load shim from this
#|                                 dir instead of the staged dir
#|   ONNX_NATIVE_WITH_CUDA=1       (deferred) opt in to CUDA provider
#|                                 on Linux — picks the gpu tarball
#|                                 variant and compiles the shim
#|                                 with ONNX_SHIM_WITH_CUDA

class Build {

	# --- Constants ------------------------------------------------------

	#| Our own GitHub Release hosts one tarball per platform,
	#| each containing libonnxruntime + a pre-built shim. Raku
	#| dist installs pull from here so users don't depend on
	#| Microsoft's CDN or have to compile a shim at install time.
	#|
	#| Override via $ONNX_NATIVE_BINARY_URL to point at Microsoft's
	#| upstream (github.com/microsoft/onnxruntime/releases/download)
	#| for dev iteration — Build.rakumod handles both layouts
	#| automatically (see !extract-archive).
	constant $DEFAULT-BASE-URL =
		'https://github.com/m-doughty/ONNX-Native/releases/download';

	#| Artifact naming: our tarballs are flat (lib/ and include/
	#| at the root), named onnxruntime-<platform>.<ext>. Microsoft's
	#| upstream wraps everything in onnxruntime-<plat>-<ver>/ and
	#| names the artifact with a version suffix; we detect and
	#| normalise both layouts at extract time.
	constant $OUR-ARTIFACT-PATTERN = 'onnxruntime-%s.%s';
	constant $UPSTREAM-ARTIFACT-PATTERN = 'onnxruntime-%s-%s.%s';

	#| Minimum glibc the prebuilt Linux archives are compatible with.
	#| Microsoft's ORT Linux prebuilts target Ubuntu 20.04's glibc
	#| 2.31 since ORT 1.18+. Bump in lockstep with upstream.
	constant $MIN-GLIBC = v2.31;

	#| Map (OS, hardware) → OUR artefact platform slug. These
	#| match the `matrix.platform` values in
	#| .github/workflows/build-binaries.yml so the tarball names
	#| our Release produces can be resolved back here.
	my %PLATFORM-SLUGS =
		'darwin-arm64'    => 'macos-arm64',
		'darwin-x86_64'   => 'macos-x86_64',
		'linux-x86_64'    => 'linux-x86_64-glibc',
		'linux-aarch64'   => 'linux-aarch64-glibc',
		'win32-x86_64'    => 'windows-x86_64',
		'mswin32-x86_64'  => 'windows-x86_64',
	;

	#| When pointing Build at Microsoft's upstream URL as a dev
	#| convenience, translate our slug to their naming. Keyed
	#| the same way as %PLATFORM-SLUGS but with Microsoft's
	#| artefact-name convention.
	my %UPSTREAM-SLUGS =
		'macos-arm64'            => 'osx-arm64',
		'macos-x86_64'           => 'osx-x86_64',
		'linux-x86_64-glibc'     => 'linux-x64',
		'linux-aarch64-glibc'    => 'linux-aarch64',
		'windows-x86_64'         => 'win-x64',
	;

	# --- Entry point ----------------------------------------------------

	method build($dist-path) {
		my Bool $prefer-system = ?%*ENV<ONNX_NATIVE_PREFER_SYSTEM>;
		my Bool $binary-only   = ?%*ENV<ONNX_NATIVE_BINARY_ONLY>;

		my Str $binary-tag = self!binary-tag($dist-path);
		my Str $plat = self!detect-platform;

		# Copy BINARY_TAG into resources so FFI.rakumod can locate
		# the staged dir at runtime. Tiny text file — survives
		# zef's resource-hashing rename intact.
		self!stage-binary-tag($dist-path);

		my IO::Path $root  = self!staged-root-dir($binary-tag);
		my IO::Path $stage = $root.add('lib');

		if $prefer-system {
			say "ONNX_NATIVE_PREFER_SYSTEM=1 — using system libonnxruntime.";
			return self!build-from-system($dist-path, $stage);
		}

		without $plat {
			note "⚠️  No prebuilt available for "
				~ "({$*KERNEL.name}-{$*KERNEL.hardware}); "
				~ "falling back to system libonnxruntime.";
			if $binary-only {
				die "ONNX_NATIVE_BINARY_ONLY=1 set but no prebuilt platform "
				  ~ "for { $*KERNEL.name }-{ $*KERNEL.hardware }.";
			}
			return self!build-from-system($dist-path, $stage);
		}

		# Guard: Microsoft's Linux prebuilts are glibc 2.31+. On
		# older systems the dylib loads but dies at first symbol
		# use with "GLIBC_2.xx not found".
		if $plat.starts-with('linux') {
			my Version $have = self!detect-glibc-version;
			if $have.defined && $have cmp $MIN-GLIBC == Less {
				if $binary-only {
					die "ONNX_NATIVE_BINARY_ONLY=1 set but system glibc "
					  ~ "$have is older than prebuilt target $MIN-GLIBC "
					  ~ "($plat / $binary-tag). Install a newer "
					  ~ "libonnxruntime or unset ONNX_NATIVE_BINARY_ONLY.";
				}
				note "⚠️  System glibc $have is older than prebuilt "
				   ~ "target $MIN-GLIBC — falling back to system "
				   ~ "libonnxruntime.";
				return self!build-from-system($dist-path, $stage);
			}
		}

		# Primary: our own GitHub Release (or user-set override).
		if self!try-prebuilt($dist-path, $plat, $binary-tag, $root) {
			say "✅ Installed prebuilt ONNX Runtime ($plat) for "
			  ~ "$binary-tag → $stage.";
			self!compile-shim($dist-path, $stage, $root.add('include'));
			return True;
		}

		# Secondary: Microsoft upstream, but only if the user didn't
		# explicitly set ONNX_NATIVE_BINARY_URL (their override wins,
		# they get what they asked for). This covers the interim
		# period before our own Release is published for a new
		# BINARY_TAG — new users get a working install immediately
		# instead of having to set an env var.
		unless %*ENV<ONNX_NATIVE_BINARY_URL> {
			note "⚠️  Prebuilt unavailable at our Release — "
			   ~ "trying Microsoft upstream.";
			my %saved = %*ENV;
			%*ENV<ONNX_NATIVE_BINARY_URL> =
				'https://github.com/microsoft/onnxruntime/releases/download';
			LEAVE %*ENV<ONNX_NATIVE_BINARY_URL> = %saved<ONNX_NATIVE_BINARY_URL>;
			if self!try-prebuilt($dist-path, $plat, $binary-tag, $root) {
				say "✅ Installed ONNX Runtime ($plat) from upstream "
				  ~ "for $binary-tag → $stage.";
				self!compile-shim($dist-path, $stage, $root.add('include'));
				return True;
			}
		}

		if $binary-only {
			die "ONNX_NATIVE_BINARY_ONLY=1 set but prebuilt download "
			  ~ "failed for $plat ($binary-tag) from all known "
			  ~ "sources (our Release + Microsoft upstream).";
		}

		note "⚠️  Prebuilt unavailable for $plat ($binary-tag) — "
		   ~ "falling back to system libonnxruntime.";
		self!build-from-system($dist-path, $stage);
	}

	# --- System-ORT fallback path ---------------------------------------

	#| Locate a system libonnxruntime + headers, compile the shim
	#| against them. Dies if we can't find a workable combination.
	method !build-from-system($dist-path, IO::Path $stage --> Bool) {
		my ($lib-dir, $include-dir) = self!find-system-ort;
		without $lib-dir {
			die "❌ No prebuilt available and no system libonnxruntime "
			  ~ "found. Install via `brew install onnxruntime`, "
			  ~ "`apt install libonnxruntime-dev`, or point "
			  ~ "ONNX_NATIVE_LIB_DIR at a directory containing "
			  ~ "libonnxruntime (the headers must also be discoverable "
			  ~ "relative to it at ../include/onnxruntime_c_api.h).";
		}
		without $include-dir {
			die "❌ Found system libonnxruntime at $lib-dir but no "
			  ~ "headers alongside it. Install the -dev package "
			  ~ "(apt: libonnxruntime-dev) or point "
			  ~ "ONNX_NATIVE_LIB_DIR at a prefix that includes headers.";
		}
		$stage.mkdir;
		self!compile-shim($dist-path, $stage, $include-dir, :$lib-dir);
		True;
	}

	#| Probe known system install locations for libonnxruntime +
	#| headers. Returns (lib-dir, include-dir) or (Nil, Nil).
	method !find-system-ort(--> List) {
		my $os = $*KERNEL.name.lc;
		my $ext = $os ~~ /darwin/ ?? 'dylib'
			   !! $*DISTRO.is-win ?? 'dll'
			   !! 'so';

		# Honour explicit override first
		with %*ENV<ONNX_NATIVE_LIB_DIR> -> $override {
			my $dir = $override.IO;
			if $dir.d {
				my $hdr = self!find-ort-header($dir);
				return ($dir, $hdr) if $hdr.defined;
				# try ../include relative to lib-dir
				my $sibling = $dir.parent.add('include');
				$hdr = self!find-ort-header($sibling);
				return ($dir, $hdr) if $hdr.defined;
			}
		}

		# Homebrew on macOS
		if $os ~~ /darwin/ {
			for </opt/homebrew/opt/onnxruntime /usr/local/opt/onnxruntime> -> $prefix {
				my $lib = $prefix.IO.add('lib');
				my $inc = self!find-ort-header($prefix.IO.add('include'));
				return ($lib, $inc) if $lib.d && $inc.defined;
			}
		}

		# Linux + generic POSIX
		for </usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu> -> $dir {
			my $lib-io = $dir.IO;
			next unless $lib-io.d;
			my $has-lib = $lib-io.dir.grep({
				my $bn = .basename;
				$bn.starts-with('libonnxruntime.')
					&& $bn.contains(".$ext");
			}).elems > 0;
			next unless $has-lib;
			# corresponding header dirs
			my $prefix = $lib-io.parent;
			my $inc = self!find-ort-header($prefix.add('include'));
			return ($lib-io, $inc) if $inc.defined;
		}

		(Nil, Nil);
	}

	method !find-ort-header(IO::Path $dir --> IO::Path) {
		return Nil unless $dir.d;
		my $direct = $dir.add('onnxruntime_c_api.h');
		return $dir if $direct.f;
		# Debian packages nest headers under onnxruntime/
		my $nested = $dir.add('onnxruntime');
		return $nested if $nested.add('onnxruntime_c_api.h').f;
		Nil;
	}

	# --- Prebuilt binary path -------------------------------------------

	method !try-prebuilt($dist-path, Str $plat, Str $binary-tag, IO::Path $root --> Bool) {
		my Str $base-url = %*ENV<ONNX_NATIVE_BINARY_URL> // $DEFAULT-BASE-URL;
		my Str $ort-version = self!ort-version-from-tag($binary-tag);
		my Bool $is-win = $plat.starts-with('windows');

		# Decide artifact + URL shape based on the base URL. When
		# pointing at Microsoft's upstream (a convenience for dev
		# iteration), switch to their naming / path scheme.
		my ($artifact, $url, $upstream) = do if $base-url.contains('microsoft/onnxruntime') {
			my Str $up-slug = %UPSTREAM-SLUGS{$plat} // $plat;
			my Str $up-ext = $is-win ?? 'zip' !! 'tgz';
			my $up = sprintf($UPSTREAM-ARTIFACT-PATTERN, $up-slug,
				$ort-version, $up-ext);
			($up, "$base-url/v$ort-version/$up", True);
		}
		else {
			my Str $ext = $is-win ?? 'zip' !! 'tar.gz';
			my $our = sprintf($OUR-ARTIFACT-PATTERN, $plat, $ext);
			($our, "$base-url/$binary-tag/$our", False);
		}

		my IO::Path $cache-dir = self!cache-dir($binary-tag);
		my IO::Path $cached = $cache-dir.add($artifact);

		unless $cached.e {
			$cache-dir.mkdir;
			say "⬇️  Fetching $artifact from $url";
			my $rc = run 'curl', '-fL', '--progress-bar',
					 '-o', $cached.Str, $url;
			unless $rc.exitcode == 0 {
				$cached.unlink if $cached.e;
				return False;
			}
		}

		# Upstream artefacts have checksums we compute per-install
		# (can't predict them); our own artefacts are verified against
		# bundled resources/checksums.txt. Skip the check in the
		# upstream-passthrough path since the human point of it is
		# dev iteration against a trusted CDN.
		unless $upstream {
			my Str $expected = self!expected-sha($dist-path, $artifact);
			without $expected {
				note "No checksum recorded for $artifact in "
					~ "resources/checksums.txt — refusing prebuilt "
					~ "(bundled checksums are a hard security boundary).";
				return False;
			}

			my Str $actual = self!sha256($cached);
			unless $actual.defined && $actual.lc eq $expected.lc {
				note "Checksum mismatch for $artifact "
					~ "(expected $expected, got {$actual // 'unknown'}).";
				$cached.unlink;
				return False;
			}
		}

		self!extract-archive($cached, $root, :$upstream);
		True;
	}

	#| Parse the Raku-side binary tag like binaries-onnxruntime-1.20.0-r1
	#| and return the upstream ORT version (1.20.0).
	method !ort-version-from-tag(Str $tag --> Str) {
		if $tag ~~ / 'onnxruntime-' (\d+ '.' \d+ '.' \d+) '-r' \d+ $/ {
			return ~$0;
		}
		die "❌ Can't parse ORT version from BINARY_TAG '$tag'. "
		  ~ "Expected format: binaries-onnxruntime-<version>-r<rev>.";
	}

	#| Extract an archive into $dest. `:$upstream` signals that the
	#| archive is Microsoft's upstream layout (wrapped in a
	#| onnxruntime-<plat>-<ver>/ directory that must be flattened)
	#| vs. our own layout (already flat at the root with lib/ and
	#| include/ directly).
	method !extract-archive(IO::Path $archive, IO::Path $dest, Bool :$upstream = False) {
		if $dest.d {
			# Blow away previous staging but keep the dir itself so
			# external tools that cached the path still find
			# something there.
			for $dest.dir -> $entry {
				$entry.d ?? run('rm', '-rf', $entry.Str) !! $entry.unlink;
			}
		}
		$dest.mkdir;

		# Extract to a tmp dir first, then move into place. This
		# keeps $dest's existence invariant while we figure out
		# the source layout.
		my IO::Path $tmp = $dest.add('.extract-tmp');
		$tmp.mkdir;
		self!run-extract($archive, $tmp);

		# Locate the root of the useful content — upstream has a
		# single wrapping dir; our flat tarballs don't. The
		# heuristic is "look for a lib/ dir": either $tmp/lib
		# (flat) or $tmp/<single-dir>/lib (wrapped).
		my IO::Path $src-root = do if $tmp.add('lib').d {
			$tmp;
		}
		else {
			my @tops = $tmp.dir.grep(*.d);
			unless @tops.elems == 1 && @tops[0].add('lib').d {
				die "❌ Unexpected archive layout in $archive — "
				  ~ "expected either a flat lib/include at root "
				  ~ "or a single wrapping dir containing them.";
			}
			@tops[0];
		}

		for <lib include> -> $subdir {
			my IO::Path $src = $src-root.add($subdir);
			next unless $src.d;   # include/ optional in our flat bundles
			rename $src, $dest.add($subdir);
		}

		run 'rm', '-rf', $tmp.Str;

		# Sanity: libonnxruntime must be present under any
		# versioned variant.
		my Str $ext = $*KERNEL.name.lc ~~ /darwin/ ?? 'dylib'
				   !! $*DISTRO.is-win ?? 'dll'
				   !! 'so';
		my @found = $dest.add('lib').dir.grep({
			my $bn = .basename;
			$bn eq "libonnxruntime.$ext"
				|| ($bn.starts-with('libonnxruntime.')
						&& $bn.contains(".$ext"))
				|| ($bn.starts-with('libonnxruntime-')
						&& $bn.ends-with(".$ext"))
				|| $bn eq "onnxruntime.$ext";     # Windows DLL
		});
		die "❌ Prebuilt archive missing libonnxruntime.$ext"
			unless @found;
	}

	#| Extract via tar (on Unix) or PowerShell Expand-Archive (on
	#| Windows, for .zip). tar's Windows behaviour parses
	#| `C:\path` as a remote host, so we explicitly avoid it
	#| on Windows ZIPs.
	method !run-extract(IO::Path $archive, IO::Path $dest) {
		if $archive.Str.ends-with('.zip') {
			my $rc = run 'powershell', '-NoProfile', '-Command',
				"Expand-Archive -LiteralPath '$archive' "
				~ "-DestinationPath '$dest' -Force";
			die "❌ Failed to extract $archive" unless $rc.exitcode == 0;
		}
		else {
			my $rc = run 'tar', '-xzf', $archive.Str, '-C', $dest.Str;
			die "❌ Failed to extract $archive." unless $rc.exitcode == 0;
		}
	}

	# --- Shim compilation ----------------------------------------------

	method !compile-shim($dist-path, IO::Path $stage, IO::Path $include-dir, :$lib-dir) {
		my Str $os = $*KERNEL.name.lc;
		my Str $ext = $os ~~ /darwin/ ?? 'dylib'
				   !! $*DISTRO.is-win ?? 'dll'
				   !! 'so';
		# Keep the `lib` prefix on every platform — MSVC is happy
		# building libfoo.dll, and it keeps FFI.rakumod's resolver
		# logic uniform across Unix and Windows.
		my Str $shim-name = "libonnx_native_shim.$ext";
		my IO::Path $shim = $stage.add($shim-name);

		# Fast path: our prebuilt bundle ships the shim already —
		# nothing to do, just confirm and move on.
		if $shim.e {
			say "✅ onnx_native_shim already present at $shim (from prebuilt).";
			return;
		}

		$stage.mkdir;

		my Str $src = "$dist-path/src/onnx_native_shim.c";
		unless $src.IO.e {
			die "❌ Shim source missing: $src";
		}
		unless $include-dir.d {
			die "❌ Shim include dir missing: { $include-dir }";
		}
		my IO::Path $link-dir = $lib-dir // $stage;
		unless $link-dir.d {
			die "❌ Library dir missing for shim link: $link-dir";
		}

		my Bool $with-cuda = ?%*ENV<ONNX_NATIVE_WITH_CUDA>;

		my @cmd;
		given $os {
			when /darwin/ {
				# -Wl,-undefined,error: default on macOS but set
				# it explicitly so a future env-level override
				# can't sneak in a lenient link.
				@cmd = 'cc', '-O2', '-fPIC', '-dynamiclib',
					'-Wall', '-Wextra', '-Wno-unused-parameter',
					'-install_name', "\@loader_path/$shim-name",
					'-I', $include-dir.Str,
					'-L', $link-dir.Str,
					'-lonnxruntime',
					'-Wl,-rpath,@loader_path',
					'-Wl,-undefined,error',
					'-o', $shim.Str, $src;
				@cmd.splice(2, 0, '-DONNX_SHIM_WITH_CUDA=1') if $with-cuda;
			}
			when $*DISTRO.is-win {
				# MSVC `cl` — expects to be invoked inside a VS Dev
				# Command Prompt (CI uses ilammy/msvc-dev-cmd).
				# Links against onnxruntime.lib (import library)
				# staged alongside onnxruntime.dll by Microsoft's
				# prebuilt. MSVC already fails the link on
				# unresolved symbols.
				my IO::Path $import-lib = $link-dir.add('onnxruntime.lib');
				unless $import-lib.e {
					die "❌ onnxruntime.lib not found at $import-lib. "
					  ~ "Microsoft's Windows prebuilt includes this "
					  ~ "import library alongside the DLL — ensure "
					  ~ "the archive extracted correctly.";
				}
				@cmd = 'cl', '/LD', '/O2', '/nologo', '/EHsc',
					"/I$include-dir",
					$src,
					'/link',
					"/LIBPATH:$link-dir",
					'onnxruntime.lib',
					"/OUT:$shim";
				@cmd.splice(4, 0, '/DONNX_SHIM_WITH_CUDA=1') if $with-cuda;
			}
			default {
				# -Wl,--no-undefined: GCC's default for `-shared`
				# is to let undefined references slide at link
				# time and fail at dlopen. That turns a missing
				# `-lonnxruntime` resolve into a runtime
				# "undefined symbol: OrtGetApiBase" at first load
				# — which is miserable to diagnose. --no-undefined
				# flips that to a hard link error instead.
				#
				# --enable-new-dtags makes the `-Wl,-rpath` we set
				# emit a RUNPATH entry (searched after LD_LIBRARY_PATH)
				# rather than the legacy RPATH (searched before).
				# Matters for users who explicitly override via
				# LD_LIBRARY_PATH — their override wins.
				@cmd = 'cc', '-O2', '-fPIC', '-shared',
					'-Wall', '-Wextra', '-Wno-unused-parameter',
					'-I', $include-dir.Str,
					'-L', $link-dir.Str,
					'-lonnxruntime',
					'-Wl,-rpath,$ORIGIN',
					'-Wl,--enable-new-dtags',
					'-Wl,--no-undefined',
					'-o', $shim.Str, $src;
				@cmd.splice(2, 0, '-DONNX_SHIM_WITH_CUDA=1') if $with-cuda;
			}
		}

		my $rc = run |@cmd, :out, :err;
		my $out = $rc.out.slurp(:close);
		my $err = $rc.err.slurp(:close);
		if $rc.exitcode == 0 {
			say "✅ Compiled onnx_native_shim → $shim.";
			# Dump the shim's linkage so a broken build in CI
			# surfaces the missing NEEDED / RPATH immediately in
			# the log rather than at first dlopen.
			self!inspect-shim($shim);
		}
		else {
			die "❌ Failed to compile onnx_native_shim: $err\nSTDOUT: $out\n";
		}
	}

	#| Print the shim's linkage info (NEEDED libraries, rpath /
	#| runpath, exported symbols) using platform-native tools.
	#| Best-effort diagnostic — swallows failures because the
	#| install path shouldn't break just because `readelf` isn't
	#| in PATH on some obscure distro.
	method !inspect-shim(IO::Path $shim) {
		my Str $os = $*KERNEL.name.lc;
		my @probes = do given $os {
			when /darwin/ {
				(('otool', '-L', $shim.Str),)
			}
			when $*DISTRO.is-win {
				(('dumpbin', '/dependents', $shim.Str),
				 ('dumpbin', '/exports',    $shim.Str),)
			}
			default {
				(('ldd', $shim.Str),
				 ('readelf', '-d', $shim.Str),)
			}
		};
		for @probes -> @cmd {
			my $proc = try { run |@cmd, :out, :err };
			next unless $proc.defined;
			my $out = $proc.out.slurp(:close);
			$proc.err.slurp(:close);
			say "--- { @cmd.join(' ') } ---";
			print $out;
		}
	}

	# --- Shared helpers -------------------------------------------------

	method !staged-root-dir(Str $binary-tag --> IO::Path) {
		my Str $base = %*ENV<ONNX_NATIVE_DATA_DIR>
			// %*ENV<XDG_DATA_HOME>
			// ($*DISTRO.is-win
					?? (%*ENV<LOCALAPPDATA>
							// "{%*ENV<USERPROFILE> // '.'}\\AppData\\Local")
					!! "{%*ENV<HOME> // '.'}/.local/share");
		"$base/ONNX-Native/$binary-tag".IO;
	}

	method !stage-binary-tag($dist-path) {
		my IO::Path $src = "$dist-path/BINARY_TAG".IO;
		my IO::Path $dst = "$dist-path/resources/BINARY_TAG".IO;
		$dst.parent.mkdir;
		copy $src, $dst;
	}

	method !cache-dir(Str $binary-tag --> IO::Path) {
		my Str $base = %*ENV<ONNX_NATIVE_CACHE_DIR>
			// %*ENV<XDG_CACHE_HOME>
			// "{%*ENV<HOME> // '.'}/.cache";
		"$base/ONNX-Native-binaries/$binary-tag".IO;
	}

	method !binary-tag($dist-path --> Str) {
		my IO::Path $file = "$dist-path/BINARY_TAG".IO;
		unless $file.e {
			die "❌ Missing BINARY_TAG file at { $file }. This file must "
			  ~ "contain the pinned binary release tag "
			  ~ "(e.g. 'binaries-onnxruntime-1.20.0-r1').";
		}
		my Str $tag = $file.slurp.trim;
		die "❌ BINARY_TAG file is empty." unless $tag.chars;
		$tag;
	}

	method !expected-sha($dist-path, Str $artifact --> Str) {
		my IO::Path $file = "$dist-path/resources/checksums.txt".IO;
		return Str unless $file.e;
		for $file.slurp.lines -> Str $line {
			my Str $trimmed = $line.trim;
			next if $trimmed eq '' || $trimmed.starts-with('#');
			my @parts = $trimmed.words;
			next unless @parts.elems >= 2;
			return @parts[0] if @parts[1] eq $artifact;
		}
		Str;
	}

	method !sha256(IO::Path $file --> Str) {
		if $*DISTRO.is-win {
			my $proc = run 'certutil', '-hashfile', $file.Str, 'SHA256',
						   :out, :err;
			my $out = $proc.out.slurp(:close);
			$proc.err.slurp(:close);
			for $out.lines -> Str $line {
				my Str $t = $line.subst(/\s+/, '', :g).lc;
				return $t if $t.chars == 64 && $t ~~ /^ <[0..9a..f]>+ $/;
			}
			return Str;
		}
		my $proc = run 'shasum', '-a', '256', $file.Str, :out, :err;
		my $out = $proc.out.slurp(:close);
		$proc.err.slurp(:close);
		$out.words.head;
	}

	method !detect-platform(--> Str) {
		my Str $key = "{$*KERNEL.name.lc}-{$*KERNEL.hardware.lc}";
		%PLATFORM-SLUGS{$key};
	}

	#| Parse `ldd --version` for the system's glibc version. Returns a
	#| Version on glibc systems, undefined Version on musl (ldd --version
	#| exits non-zero) or when ldd is absent / unparseable. Only
	#| meaningful on Linux — don't call on other OSes.
	method !detect-glibc-version(--> Version) {
		my $proc = try { run 'ldd', '--version', :out, :err };
		return Version without $proc;
		my $out = $proc.out.slurp(:close);
		$proc.err.slurp(:close);
		return Version unless $proc.exitcode == 0;
		my $first = $out.lines.head // '';
		if $first ~~ / (\d+ '.' \d+ [ '.' \d+ ]?) \s* $ / {
			return Version.new(~$0);
		}
		Version;
	}
}
