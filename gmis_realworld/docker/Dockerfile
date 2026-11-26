# ATEC2026 on-board runtime — A2 Pro + PiperX on the mini-PC.
#
# Why a container at all, when the target is one machine we control: because the rules lock
# the code and configuration at inspection and require it to be escrowed, and "it worked on
# the machine we set up in August" is not a thing that can be re-created from a git tag.  An
# image digest can.  It also makes the mini-PC re-flashable in minutes rather than in a
# morning of `setup_minipc.sh`.
#
# Two-stage: the builder carries a compiler (CycloneDDS does not always ship a usable wheel)
# and the runtime carries none of it.
#
# CPU ONLY, deliberately.  The target has no GPU, the policies are three-layer MLPs (146 us
# per leg step), and the default torch wheel would drag ~2.5 GB of CUDA into an image that
# has to fit on, and be copied to, a 512 GB mini-PC.

# ---------------------------------------------------------------------------- builder
# The target is an x86_64 mini-PC and the workstation that builds the escrow image may not
# be one.  `--platform linux/amd64` is passed by docker/build.sh and by compose.yml's
# `platforms:`; the runtime stage then ASSERTS the architecture, so a build that slipped
# through by another route fails loudly instead of producing an image that loads, passes its
# own gate, and then will not run on the robot.
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel setuptools

# torch first and explicitly from the CPU index -- see the header.
ARG TORCH_VERSION=2.5.1
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

# ONE source of truth for the dependency set.  This used to COPY the requirements file and
# then install a hand-maintained copy of its contents, so the host installer and the image
# could drift apart silently -- for code the rules lock at inspection, that is a defect.
# torch is filtered out because it is already installed above from the CPU index; letting
# pip resolve `torch==` against the default index here would pull the ~2.5 GB CUDA build.
COPY onboard/requirements-onboard.txt /tmp/requirements-onboard.txt
RUN grep -vE '^[[:space:]]*(#|$)' /tmp/requirements-onboard.txt \
      | grep -vE '^torch==' > /tmp/req-nontorch.txt \
    && echo "installing:" && cat /tmp/req-nontorch.txt \
    && pip install -r /tmp/req-nontorch.txt

# --- robot SDKs ------------------------------------------------------------------------
# Optional at BUILD time so the image can still be produced (and the whole stack verified)
# on a machine that cannot reach these sources.  `tools/doctor.py` reports them as MISSING
# if they are absent, which is exactly the signal we want rather than a build that silently
# ships a broken deployment image.
ARG WITH_SDK=1

# CycloneDDS: the python binding at the version unitree_sdk2py pins is SOURCE ONLY -- there
# is no 0.10.2 wheel, and pip's failure message ("Could not locate cyclonedds") describes
# the symptom rather than the cause.  The C library has to exist and CYCLONEDDS_HOME has to
# point at it BEFORE pip runs.  Getting that order wrong is the single most common way this
# install fails, which is why it is a separate, explicit step.
ENV CYCLONEDDS_HOME=/opt/cyclonedds
# The prefix is created UNCONDITIONALLY: the runtime stage COPYs /opt/cyclonedds without a
# guard, so with WITH_SDK=0 -- the offline/CI mode build.sh documents -- the build used to
# fail on a missing COPY source rather than produce the SDK-less image it promises.
RUN mkdir -p ${CYCLONEDDS_HOME}
# Pinned to the TAG that matches the pinned python binding (cyclonedds==0.10.2, tag 0.10.2
# = commit 9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd).  This was `-b releases/0.10.x`, a
# moving branch: two builds of the same git rev could contain different C libraries, which
# is precisely what an escrowed digest exists to rule out.
ARG CYCLONEDDS_REF=0.10.2
RUN if [ "$WITH_SDK" = "1" ]; then \
      git clone --depth 1 -b "${CYCLONEDDS_REF}" \
        https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cdds \
      && cmake -S /tmp/cdds -B /tmp/cdds/build \
           -DCMAKE_INSTALL_PREFIX=${CYCLONEDDS_HOME} \
           -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
      && cmake --build /tmp/cdds/build --target install -j"$(nproc)" \
      && rm -rf /tmp/cdds ; \
    fi

# The three robot SDKs.  unitree_sdk2_python has NO tags, so the only way to pin it is the
# commit; without a pin an escrowed image cannot be rebuilt from its own recipe.
ARG UNITREE_SDK_REF=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
RUN if [ "$WITH_SDK" = "1" ]; then \
      pip install "cyclonedds==0.10.2" || echo "SDK-SKIP cyclonedds" >> /opt/venv/SDK_STATUS ; \
      pip install "python-can>=4.0"    || echo "SDK-SKIP python-can" >> /opt/venv/SDK_STATUS ; \
      pip install piper_sdk            || echo "SDK-SKIP piper_sdk" >> /opt/venv/SDK_STATUS ; \
      ( git init -q /tmp/usdk \
        && git -C /tmp/usdk remote add origin \
             https://github.com/unitreerobotics/unitree_sdk2_python \
        && git -C /tmp/usdk fetch -q --depth 1 origin "${UNITREE_SDK_REF}" \
        && git -C /tmp/usdk checkout -q FETCH_HEAD \
        && pip install /tmp/usdk \
        && SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
        && cp -r /tmp/usdk/unitree_sdk2py/utils/lib "$SP/unitree_sdk2py/utils/lib" \
      ) || echo "SDK-SKIP unitree_sdk2py" >> /opt/venv/SDK_STATUS ; \
      rm -rf /tmp/usdk ; \
    else echo "SDK-SKIP all (WITH_SDK=0)" >> /opt/venv/SDK_STATUS ; fi
RUN touch /opt/venv/SDK_STATUS && pip freeze > /opt/venv/SDK_LOCK

# unitree_sdk2py's setup.py is `find_packages()` with NO package_data, so the two prebuilt
# CRC shared objects under unitree_sdk2py/utils/lib/ are not installed by pip -- the `cp`
# above restores them.  Without them ChannelFactoryInitialize dies with "crc_amd64.so:
# cannot open shared object file", and that is the FIRST thing every real-backend command
# does: `run --backend real`, `prep`, `estop`, `motormap`.  The image built, passed its own
# acceptance gate, and could not talk to the robot at all.  Assert it here, so that failure
# lands on a workstation instead of on the course.
RUN if [ "$WITH_SDK" = "1" ]; then \
      python -c "from unitree_sdk2py.utils.crc import CRC; CRC(); print('unitree CRC lib ok')" ; \
    fi

# unitree_sdk2py depends on `opencv-python` -- unpinned, and the GUI build.  Installing it
# leaves TWO opencvs in the venv, `import cv2` resolves to whichever landed last, and the
# version floats between builds.  For code that the rules lock at inspection and require to
# be escrowed, a transitive dependency that changes version on its own is a defect, not an
# inconvenience: the image you tested is not the image you ship.
RUN pip uninstall -y opencv-python 2>/dev/null || true; \
    pip install --force-reinstall --no-deps "opencv-python-headless==4.10.0.84"

# Refuse to produce an image whose installed versions disagree with the pins.  A build that
# passes this is one whose dependency set can be stated exactly in the escrow submission.
COPY onboard/requirements-onboard.txt /tmp/req.txt
RUN python - <<'PIN'
import re, subprocess, sys
# Normalise the way PEP 503 does: distribution names are case-insensitive and treat
# -/_/. as equivalent, and a local version segment (torch's "+cpu") is a build tag on the
# same release, not a different one.  Without both rules this check cries wolf on PyYAML
# and on the very CPU wheel it is supposed to be enforcing.
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
base = lambda v: v.split("+", 1)[0]
pins = {norm(k): v for k, v in re.findall(
    r"^([A-Za-z0-9_.-]+)==([\d.]+)", open("/tmp/req.txt").read(), re.M)}
have = {norm(k): v for k, v in (
    l.split("==", 1) for l in subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True).stdout.splitlines() if "==" in l)}
bad = [(k, v, have.get(k)) for k, v in pins.items() if base(have.get(k, "")) != v]
for k, want, got in bad:
    print(f"PIN DRIFT: {k} pinned {want}, installed {got}", file=sys.stderr)
print(f"pins checked: {len(pins)}, drift: {len(bad)}")
sys.exit(1 if bad else 0)
PIN

# ---------------------------------------------------------------------------- runtime
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # The stack sets these itself in onboard/__init__.py; declaring them here too means a
    # bare `python -c` inside the container behaves like the mission does.
    ONBOARD_TORCH_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MISSION_REAL_NOISE=0

# can-utils for `candump`/`cansend` when the arm misbehaves; iproute2 to read link state.
# libGL/glib are opencv's runtime deps even in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        can-utils iproute2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
# The python binding is a thin ctypes/cython layer over libddsc.so; without the C library in
# the runtime stage it imports and then fails at the first channel with a loader error.
COPY --from=builder /opt/cyclonedds /opt/cyclonedds
# Set absolutely, not appended: python:3.11-slim-bookworm does not define LD_LIBRARY_PATH,
# so `:${LD_LIBRARY_PATH}` only ever contributed a trailing colon and a build warning.
ENV CYCLONEDDS_HOME=/opt/cyclonedds \
    LD_LIBRARY_PATH=/opt/cyclonedds/lib

WORKDIR /app
# ONLY what the runtime imports.  Verified by walking the import graph: 13 MB / 90 files, of
# which 12 MB is the policy weights.  The ROOT .dockerignore whitelists exactly this set;
# without it the context is the whole 485 MB repo.
COPY onboard/                      /app/onboard/
COPY course_mission/__init__.py    /app/course_mission/__init__.py
COPY course_mission/armkin.py      /app/course_mission/armkin.py
COPY course_mission/catalogue.py   /app/course_mission/catalogue.py
COPY course_mission/deploy.py      /app/course_mission/deploy.py
COPY course_mission/mathx.py       /app/course_mission/mathx.py
COPY course_mission/navfuse.py     /app/course_mission/navfuse.py
COPY course_mission/perception.py  /app/course_mission/perception.py
COPY eunha_rl/epkg/rma.py          /app/eunha_rl/epkg/rma.py
COPY docker/entrypoint.sh          /usr/local/bin/entrypoint

# Provenance, for the escrow and for the run report.  Passed in by build.sh.
ARG GIT_REV=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="atec-onboard" \
      org.opencontainers.image.description="ATEC2026 A2 Pro + PiperX on-board runtime (CPU)" \
      org.opencontainers.image.revision="${GIT_REV}" \
      org.opencontainers.image.created="${BUILD_DATE}"
RUN printf 'git_rev=%s\nbuild_date=%s\n' "${GIT_REV}" "${BUILD_DATE}" > /app/BUILD_INFO \
    && chmod +x /usr/local/bin/entrypoint

# The stack resolves its output paths relative to the repo root -- onboard/logs/<run>/ for
# reports and the ledger, onboard/logs/calib/ for checkerboard views.  Inside the image that
# is /app/onboard/logs, and docker/run.sh binds the host directory there.  /logs is kept as
# a symlink because the documented parity command writes to it.
RUN mkdir -p /app/onboard/logs && ln -sfn /app/onboard/logs /logs

# `docker stop` must reach python, which handles SIGTERM by damping the legs (onboard/run.py).
# Without this the default is still SIGTERM, but stating it keeps the contract with
# docker/run.sh's --stop-signal/--stop-timeout visible in one place.
STOPSIGNAL SIGTERM

# The robot's mini-PC is an Intel N150.  An image built on another architecture would load
# and pass every check below on the build host and then fail to start on the robot.
RUN test "$(uname -m)" = "x86_64" || { \
      echo "IMAGE ARCH $(uname -m) != x86_64 -- rebuild with --platform linux/amd64" >&2; \
      exit 1; }

# Fail the build if the stack cannot even import without a simulator.  A deployment image
# that does not pass its own acceptance gate should not exist.
RUN python /app/onboard/tools/offsim_gate.py

ENTRYPOINT ["/usr/local/bin/entrypoint"]
CMD ["doctor"]
