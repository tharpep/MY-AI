"""
Model Manager CLI
Simple command-line tool for managing tuned model versions
"""

import argparse
import sys
from pathlib import Path
from .model_registry import ModelRegistry, get_model_registry
from core.config import get_config


def list_versions():
    """List all model versions"""
    config = get_config()
    registry = get_model_registry(config)
    registry.list_versions()


def set_active_version(version: str):
    """Set a version as active"""
    config = get_config()
    registry = get_model_registry(config)
    
    if registry.set_active_version(version):
        print(f"✅ Version {version} is now active")
    else:
        print(f"❌ Failed to set version {version} as active")
        sys.exit(1)


def get_active_version():
    """Get the currently active version"""
    config = get_config()
    registry = get_model_registry(config)
    
    active_version = registry.get_active_version()
    if active_version:
        print(f"Active version: {active_version.version}")
        print(f"Created: {active_version.created_at}")
        print(f"Model: {active_version.model_name}")
        if active_version.notes:
            print(f"Notes: {active_version.notes}")
    else:
        print("No active version found")


def get_latest_version():
    """Get the latest version"""
    config = get_config()
    registry = get_model_registry(config)
    
    latest_version = registry.get_latest_version()
    if latest_version:
        print(f"Latest version: {latest_version.version}")
        print(f"Created: {latest_version.created_at}")
        print(f"Model: {latest_version.model_name}")
        if latest_version.notes:
            print(f"Notes: {latest_version.notes}")
    else:
        print("No versions found")


def show_version_info(version: str):
    """Show detailed info for a specific version"""
    config = get_config()
    registry = get_model_registry(config)
    
    version_obj = registry.get_version(version)
    if not version_obj:
        print(f"❌ Version {version} not found")
        sys.exit(1)
    
    print(f"\n=== Version {version_obj.version} ===")
    print(f"Model: {version_obj.model_name}")
    print(f"Base model: {version_obj.base_model}")
    print(f"Created: {version_obj.created_at}")
    print(f"Training epochs: {version_obj.training_epochs}")
    print(f"Batch size: {version_obj.batch_size}")
    print(f"Learning rate: {version_obj.learning_rate}")
    print(f"Device: {version_obj.device}")
    print(f"Active: {'Yes' if version_obj.is_active else 'No'}")
    
    if version_obj.training_time_seconds:
        print(f"Training time: {version_obj.training_time_seconds:.1f} seconds")
    if version_obj.final_loss:
        print(f"Final loss: {version_obj.final_loss:.4f}")
    if version_obj.model_size_mb:
        print(f"Model size: {version_obj.model_size_mb:.1f} MB")
    if version_obj.notes:
        print(f"Notes: {version_obj.notes}")
    
    # Show file path
    version_path = registry.get_version_path(version)
    if version_path:
        print(f"Path: {version_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Model Version Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List versions
    subparsers.add_parser('list', help='List all model versions')
    
    # Set active version
    set_parser = subparsers.add_parser('set-active', help='Set a version as active')
    set_parser.add_argument('version', help='Version to set as active (e.g., v1.0)')
    
    # Get active version
    subparsers.add_parser('active', help='Show the currently active version')
    
    # Get latest version
    subparsers.add_parser('latest', help='Show the latest version')
    
    # Show version info
    info_parser = subparsers.add_parser('info', help='Show detailed info for a version')
    info_parser.add_argument('version', help='Version to show info for (e.g., v1.0)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_versions()
        elif args.command == 'set-active':
            set_active_version(args.version)
        elif args.command == 'active':
            get_active_version()
        elif args.command == 'latest':
            get_latest_version()
        elif args.command == 'info':
            show_version_info(args.version)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
