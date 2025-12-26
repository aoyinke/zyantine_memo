"""
自衍体AI系统主入口
"""
import sys
import argparse
import json
import traceback
import logging
from typing import Optional

# 添加项目根目录到Python路径
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zyantine_facade import ZyantineFacade, create_zyantine

# 导入工具模块
from utils.logger import get_logger
from utils.error_handler import handle_error, register_shutdown_handler, GracefulShutdown

logger = get_logger("main")
shutdown_handler = GracefulShutdown()


def load_profile(file_path: str) -> Optional[dict]:
    """加载配置文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                # 尝试其他格式
                import yaml
                return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None


def interactive_mode(facade: ZyantineFacade):
    """交互模式"""
    print("\n" + "=" * 60)
    print("自衍体AI交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'status' 查看系统状态")
    print("输入 'save' 保存记忆")
    print("输入 'cleanup' 清理记忆")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式...")
                break
            elif user_input.lower() == 'status':
                status = facade.get_status()
                print(f"\n系统状态:")
                print(f"  会话ID: {status['session_id']}")
                print(f"  记忆总数: {status['memory_stats'].get('total_memories', 0)}")
                print(f"  对话历史: {status['conversation_history_length']} 条")
                print(f"  处理模式: {status['processing_mode']}")
                print()
            elif user_input.lower() == 'save':
                success = facade.save_memory()
                print(f"保存记忆: {'成功' if success else '失败'}")
            elif user_input.lower() == 'cleanup':
                success = facade.cleanup()
                print(f"清理记忆: {'成功' if success else '失败'}")
            elif user_input:
                response = facade.chat(user_input)
                print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\n\n检测到中断，正在关闭系统...")
            break
        except Exception as e:
            logger.error(f"处理错误: {e}")
            print(f"处理错误: {e}")


def batch_mode(facade: ZyantineFacade, input_file: str, output_file: str):
    """批量处理模式"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            inputs = [line.strip() for line in f if line.strip()]

        logger.info(f"批量处理 {len(inputs)} 条输入...")

        results = []
        for i, user_input in enumerate(inputs, 1):
            logger.info(f"处理第 {i}/{len(inputs)} 条: {user_input[:50]}...")

            try:
                response = facade.chat(user_input)
                results.append({
                    "input": user_input,
                    "response": response,
                    "index": i
                })
            except Exception as e:
                logger.error(f"处理失败: {e}")
                results.append({
                    "input": user_input,
                    "error": str(e),
                    "index": i
                })

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"批量处理完成，结果已保存到: {output_file}")

    except Exception as e:
        logger.error(f"批量处理失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自衍体AI系统")

    # 配置参数
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--api-key', '-k', help='OpenAI API密钥')
    parser.add_argument('--session', '-s', default='default', help='会话ID')

    # 模式参数
    parser.add_argument('--interactive', '-i', action='store_true', help='交互模式')
    parser.add_argument('--batch', '-b', help='批量处理输入文件')
    parser.add_argument('--output', '-o', help='批量处理输出文件')

    # 其他参数
    parser.add_argument('--profile', '-p', help='用户配置文件')
    parser.add_argument('--self-profile', '-sp', help='自衍体配置文件')
    parser.add_argument('--status', action='store_true', help='显示系统状态')
    parser.add_argument('--save', action='store_true', help='保存记忆系统')
    parser.add_argument('--cleanup', action='store_true', help='清理记忆')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')

    args = parser.parse_args()

    try:
        # 设置日志级别
        log_level = getattr(logging, args.log_level)
        logger.setLevel(log_level)

        # 设置信号处理器
        shutdown_handler.setup_signal_handlers()

        logger.info(f"启动自衍体AI系统，日志级别: {args.log_level}")

        # 加载配置文件
        user_profile = load_profile(args.profile) if args.profile else None
        self_profile = load_profile(args.self_profile) if args.self_profile else None

        # 创建系统实例
        if args.api_key and not args.config:
            logger.info("使用API密钥创建系统实例")
            facade = create_zyantine(
                api_key=args.api_key,
                session_id=args.session
            )
        else:
            logger.info(f"使用配置文件创建系统实例: {args.config}")
            facade = ZyantineFacade(
                config_path=args.config,
                user_profile=user_profile,
                self_profile=self_profile,
                session_id=args.session
            )

        # 注册关闭处理器
        register_shutdown_handler(facade.shutdown, name="facade_shutdown")

        # 执行命令
        if args.status:
            status = facade.get_status()
            print(json.dumps(status, ensure_ascii=False, indent=2))

        elif args.save:
            success = facade.save_memory()
            print(f"保存记忆: {'成功' if success else '失败'}")

        elif args.cleanup:
            success = facade.cleanup()
            print(f"清理记忆: {'成功' if success else '失败'}")

        elif args.batch:
            if not args.output:
                logger.error("批量处理需要指定输出文件")
                print("错误：批量处理需要指定输出文件 (--output)")
                sys.exit(1)
            batch_mode(facade, args.batch, args.output)

        elif args.interactive or (not args.status and not args.save and not args.cleanup and not args.batch):
            interactive_mode(facade)

        # 关闭系统
        facade.shutdown()
        logger.info("系统已安全关闭")

    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        shutdown_handler.shutdown("用户中断")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        handle_error(e, raise_again=True)
    finally:
        # 确保优雅关闭
        shutdown_handler.shutdown("程序结束")


if __name__ == "__main__":
    main()