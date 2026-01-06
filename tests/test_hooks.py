import pytest
import os
import subprocess
import shutil

def test_neovim_script_generation():
    test_dir = "/tmp/test_neovim_hooks"
    os.makedirs(test_dir, exist_ok=True)
    dest_path = os.path.join(test_dir, "ai-mem.lua")
    script_path = os.path.abspath("scripts/install-neovim.sh")
    
    # Run script
    subprocess.run([script_path, "--dest", dest_path], check=True)
    
    # Check if file exists
    assert os.path.exists(dest_path)
    
    # Check content
    with open(dest_path, "r") as f:
        content = f.read()
        assert "local M = {}" in content
        assert "function M.session_start()" in content
        assert "vim.api.nvim_create_user_command" in content
    
    shutil.rmtree(test_dir)
