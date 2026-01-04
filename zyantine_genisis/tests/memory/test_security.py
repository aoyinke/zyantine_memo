"""
记忆安全管理器测试
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.memory_manager import (
    MemoryManager,
    MemoryType,
    MemoryPriority,
    MemorySecurityLevel,
    MemoryPrivacyPolicy
)
from memory.memory_exceptions import MemoryStorageError


def test_security_level_setting():
    """测试安全级别设置"""
    print("\n" + "="*60)
    print("测试1: 安全级别设置")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加一个记忆
        memory_id = manager.add_memory(
            content="这是一个测试记忆",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True  # 跳过测试不算失败
    
    # 设置安全级别
    success = manager.security_manager.set_memory_security_level(
        memory_id,
        MemorySecurityLevel.CONFIDENTIAL
    )
    
    # 获取安全级别
    security_level = manager.security_manager.get_security_level(memory_id)
    
    if success and security_level == MemorySecurityLevel.CONFIDENTIAL:
        print("  ✓ 安全级别设置测试成功")
        print(f"    记忆ID: {memory_id}")
        print(f"    安全级别: {security_level.value}")
        return True
    else:
        print("  ✗ 安全级别设置测试失败")
        return False


def test_privacy_policy_setting():
    """测试隐私策略设置"""
    print("\n" + "="*60)
    print("测试2: 隐私策略设置")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加一个记忆
        memory_id = manager.add_memory(
            content="这是一个测试记忆",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True
    
    # 设置隐私策略
    success = manager.security_manager.set_memory_privacy_policy(
        memory_id,
        MemoryPrivacyPolicy.ALLOW_OWNER
    )
    
    # 获取隐私策略
    privacy_policy = manager.security_manager.get_privacy_policy(memory_id)
    
    if success and privacy_policy == MemoryPrivacyPolicy.ALLOW_OWNER:
        print("  ✓ 隐私策略设置测试成功")
        print(f"    记忆ID: {memory_id}")
        print(f"    隐私策略: {privacy_policy.value}")
        return True
    else:
        print("  ✗ 隐私策略设置测试失败")
        return False


def test_access_control():
    """测试访问控制"""
    print("\n" + "="*60)
    print("测试3: 访问控制")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加一个记忆
        memory_id = manager.add_memory(
            content="这是一个测试记忆",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True
    
    # 设置隐私策略为仅所有者可访问
    manager.security_manager.set_memory_privacy_policy(
        memory_id,
        MemoryPrivacyPolicy.ALLOW_OWNER
    )
    
    # 授予用户访问权限
    user_id = "test_user_123"
    grant_success = manager.security_manager.grant_access(
        memory_id,
        user_id,
        "read"
    )
    
    # 检查访问权限
    has_access = manager.security_manager.check_access(
        memory_id,
        user_id,
        "read"
    )
    
    # 撤销访问权限
    revoke_success = manager.security_manager.revoke_access(
        memory_id,
        user_id
    )
    
    # 再次检查访问权限
    has_access_after_revoke = manager.security_manager.check_access(
        memory_id,
        user_id,
        "read"
    )
    
    if grant_success and has_access and revoke_success and not has_access_after_revoke:
        print("  ✓ 访问控制测试成功")
        print(f"    记忆ID: {memory_id}")
        print(f"    用户ID: {user_id}")
        print(f"    授权成功: {grant_success}")
        print(f"    授权后有访问权限: {has_access}")
        print(f"    撤销成功: {revoke_success}")
        print(f"    撤销后无访问权限: {not has_access_after_revoke}")
        return True
    else:
        print("  ✗ 访问控制测试失败")
        print(f"    授权成功: {grant_success}")
        print(f"    授权后有访问权限: {has_access}")
        print(f"    撤销成功: {revoke_success}")
        print(f"    撤销后无访问权限: {not has_access_after_revoke}")
        return False


def test_sensitive_data_detection():
    """测试敏感数据检测"""
    print("\n" + "="*60)
    print("测试4: 敏感数据检测")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加包含敏感信息的记忆
        memory_id = manager.add_memory(
            content="我的密码是123456，手机号是13812345678",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True
    
    # 获取记忆记录
    record = manager.cache.get(memory_id)
    
    if record:
        content = record.content
        # 检查是否进行了脱敏
        has_masked_phone = "****" in content
        has_masked_email = "***@" in content
        
        # 检查元数据中的敏感词
        has_sensitive_keywords = "sensitive_keywords" in record.metadata
        
        # 检查安全级别是否被设置为CONFIDENTIAL
        security_level = manager.security_manager.get_security_level(memory_id)
        is_confidential = security_level == MemorySecurityLevel.CONFIDENTIAL
        
        if has_masked_phone and is_confidential:
            print("  ✓ 敏感数据检测测试成功")
            print(f"    记忆ID: {memory_id}")
            print(f"    脱敏后内容: {content}")
            print(f"    敏感词: {record.metadata.get('sensitive_keywords', [])}")
            print(f"    手机号已脱敏: {has_masked_phone}")
            print(f"    安全级别: {security_level.value}")
            return True
        else:
            print("  ✗ 敏感数据检测测试失败")
            print(f"    手机号已脱敏: {has_masked_phone}")
            print(f"    邮箱已脱敏: {has_masked_email}")
            print(f"    检测到敏感词: {has_sensitive_keywords}")
            print(f"    安全级别: {security_level.value}")
            return False
    else:
        print("  ✗ 敏感数据检测测试失败: 无法获取记忆记录")
        return False


def test_encryption_decryption():
    """测试加密和解密"""
    print("\n" + "="*60)
    print("测试5: 加密和解密")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加一个记忆
        memory_id = manager.add_memory(
            content="这是一个需要加密的敏感信息",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True
    
    # 获取原始内容
    record = manager.cache.get(memory_id)
    original_content = record.content if record else ""
    
    # 授予用户访问权限
    user_id = manager.user_id
    manager.security_manager.grant_access(
        memory_id,
        user_id,
        "read"
    )
    
    # 加密内容
    encrypted_content = manager.security_manager.encrypt_memory(
        memory_id,
        original_content
    )
    
    # 解密内容
    decrypted_content = manager.security_manager.decrypt_memory(
        memory_id,
        encrypted_content,
        user_id
    )
    
    if original_content == decrypted_content and encrypted_content != original_content:
        print("  ✓ 加密和解密测试成功")
        print(f"    记忆ID: {memory_id}")
        print(f"    原始内容: {original_content}")
        print(f"    加密内容: {encrypted_content[:50]}...")
        print(f"    解密内容: {decrypted_content}")
        print(f"    加密前后内容不同: {encrypted_content != original_content}")
        print(f"    解密后内容一致: {original_content == decrypted_content}")
        return True
    else:
        print("  ✗ 加密和解密测试失败")
        print(f"    加密前后内容不同: {encrypted_content != original_content}")
        print(f"    解密后内容一致: {original_content == decrypted_content}")
        return False


def test_audit_log():
    """测试审计日志"""
    print("\n" + "="*60)
    print("测试6: 审计日志")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加一个记忆
        memory_id = manager.add_memory(
            content="这是一个测试记忆",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True
    
    # 执行一些操作
    manager.security_manager.set_memory_security_level(
        memory_id,
        MemorySecurityLevel.CONFIDENTIAL
    )
    
    manager.security_manager.set_memory_privacy_policy(
        memory_id,
        MemoryPrivacyPolicy.ALLOW_OWNER
    )
    
    # 获取审计日志
    audit_logs = manager.security_manager.get_audit_log(memory_id)
    
    if len(audit_logs) >= 2:
        print("  ✓ 审计日志测试成功")
        print(f"    记忆ID: {memory_id}")
        print(f"    审计日志数量: {len(audit_logs)}")
        print(f"    日志示例:")
        for i, log in enumerate(audit_logs[:2]):
            print(f"      {i+1}. {log['action']} at {log['timestamp']}")
        return True
    else:
        print("  ✗ 审计日志测试失败")
        print(f"    审计日志数量: {len(audit_logs)}")
        return False


def test_security_stats():
    """测试安全统计"""
    print("\n" + "="*60)
    print("测试7: 安全统计")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加多个记忆
        memory_ids = []
        for i in range(10):
            memory_id = manager.add_memory(
                content=f"测试记忆 {i}",
                memory_type=MemoryType.CONVERSATION,
                tags=["测试"],
                priority=MemoryPriority.MEDIUM
            )
            memory_ids.append(memory_id)
            
            # 设置不同的安全级别
            if i % 3 == 0:
                manager.security_manager.set_memory_security_level(
                    memory_id,
                    MemorySecurityLevel.CONFIDENTIAL
                )
            elif i % 3 == 1:
                manager.security_manager.set_memory_security_level(
                    memory_id,
                    MemorySecurityLevel.SECRET
                )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True
    
    # 获取安全统计
    stats = manager.security_manager.get_security_stats()
    
    if stats["total_memories_with_security"] >= 10:
        print("  ✓ 安全统计测试成功")
        print(f"    总记忆数（带安全级别）: {stats['total_memories_with_security']}")
        print(f"    总记忆数（带隐私策略）: {stats['total_memories_with_privacy']}")
        print(f"    安全级别分布: {stats['security_level_distribution']}")
        print(f"    隐私策略分布: {stats['privacy_policy_distribution']}")
        print(f"    审计日志大小: {stats['audit_log_size']}")
        print(f"    加密启用: {stats['encryption_enabled']}")
        print(f"    访问控制启用: {stats['access_control_enabled']}")
        print(f"    审计日志启用: {stats['audit_log_enabled']}")
        return True
    else:
        print("  ✗ 安全统计测试失败")
        print(f"    总记忆数: {stats['total_memories_with_security']}")
        return False


def test_auto_security_level():
    """测试自动安全级别设置"""
    print("\n" + "="*60)
    print("测试8: 自动安全级别设置")
    print("="*60)
    
    manager = MemoryManager()
    
    try:
        # 添加包含敏感词的记忆
        memory_id1 = manager.add_memory(
            content="我的密码是123456",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
        
        # 添加不包含敏感词的记忆
        memory_id2 = manager.add_memory(
            content="今天天气很好",
            memory_type=MemoryType.CONVERSATION,
            tags=["测试"],
            priority=MemoryPriority.MEDIUM
        )
    except MemoryStorageError as e:
        print(f"  ⚠ 测试跳过: API不可用 - {str(e)[:100]}")
        return True
    
    # 获取安全级别
    security_level1 = manager.security_manager.get_security_level(memory_id1)
    security_level2 = manager.security_manager.get_security_level(memory_id2)
    
    if security_level1 == MemorySecurityLevel.CONFIDENTIAL and security_level2 == MemorySecurityLevel.INTERNAL:
        print("  ✓ 自动安全级别设置测试成功")
        print(f"    包含敏感词的记忆ID: {memory_id1}")
        print(f"    安全级别: {security_level1.value}")
        print(f"    不包含敏感词的记忆ID: {memory_id2}")
        print(f"    安全级别: {security_level2.value}")
        return True
    else:
        print("  ✗ 自动安全级别设置测试失败")
        print(f"    包含敏感词的记忆安全级别: {security_level1.value}")
        print(f"    不包含敏感词的记忆安全级别: {security_level2.value}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("记忆安全管理器测试套件")
    print("="*60)
    
    tests = [
        test_security_level_setting,
        test_privacy_policy_setting,
        test_access_control,
        test_sensitive_data_detection,
        test_encryption_decryption,
        test_audit_log,
        test_security_stats,
        test_auto_security_level
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print(f"测试完成: 通过 {sum(results)}/{len(results)}, 失败 {len(results) - sum(results)}/{len(results)}")
    print("="*60)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
