#!/bin/bash

# Define variables
NAS_IP="155.98.9.165"
SHARE_NAME="mobl-nas"
MOUNT_POINT="$HOME/mnt/mobl-nas"
USERNAME="aedwards"
WORKSPACE_LINK_NAME="Datasets_NAS"

# Create mount point if it doesn't exist
mkdir -p "$MOUNT_POINT"

echo "Attempting to mount NAS..."
echo "You may be prompted for your sudo password and then the NAS password for user '$USERNAME'."

# Mount command using absolute path to id to avoid OpenSim alias
sudo mount -t cifs "//$NAS_IP/$SHARE_NAME" "$MOUNT_POINT" -o "username=$USERNAME,uid=$(/usr/bin/id -u),gid=$(/usr/bin/id -g),file_mode=0777,dir_mode=0777"

if [ $? -eq 0 ]; then
    echo "✅ NAS mounted successfully at $MOUNT_POINT"
    
    # Create symlink in the current workspace
    TARGET_DIR="$MOUNT_POINT/MOBL_shared/Datasets"
    if [ -d "$TARGET_DIR" ]; then
        ln -sfn "$TARGET_DIR" "$WORKSPACE_LINK_NAME"
        echo "✅ Created symlink '$WORKSPACE_LINK_NAME' -> $TARGET_DIR"
    else
        echo "⚠️  Warning: Target directory $TARGET_DIR not found on NAS."
        echo "Listing contents of $MOUNT_POINT to help you find the right path:"
        ls -F "$MOUNT_POINT"
    fi
else
    echo "❌ Failed to mount NAS. Please check your credentials and network connection."
fi
