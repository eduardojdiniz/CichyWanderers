#!/usr/bin/env bash
cd $(cd -P -- "$(dirname -- "$0")" && pwd -P)

export NB_UID=1000
export NB_GID=100

set_permission(){
    local folder=$1
    # make it owned by the gid of the notebook containers.
    chown -R $NB_UID:$NB_GID "${folder}"
    # make it group-setgid-writable
    chmod -R g+rws "${folder}"
    # make it user-readable-writable-executable
    chmod -R u+rwx "${folder}"
    # set the default permissions for new files to group-writable
    setfacl -d -m g::rwx "${folder}"
    setfacl -d -m u::rwx "${folder}"

    unset folder
}

CichyWanderers="/home/dinize@acct.upmchs.net/proj/CichyWanderers" && \
    set_permission $CichyWanderers
