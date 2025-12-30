"""
WebSocket 消息调试脚本
用于诊断消息传输过程中的编码和格式问题
"""
import asyncio
import json
import websockets
from typing import Optional


class MessageDebugger:
    """消息调试器"""

    def __init__(self, uri: str, client_id: str = "debug_client"):
        self.uri = uri
        self.client_id = client_id
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """连接到服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            print(f"\n{'='*60}")
            print(f"已连接到服务器: {self.uri}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    async def send_and_receive(self, payload: dict, description: str = ""):
        """发送消息并接收响应"""
        if not self.websocket:
            print("未连接到服务器")
            return None

        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

        print("\n[客户端] 准备发送消息:")
        print(f"  Payload 类型: {type(payload)}")
        print(f"  Payload 内容: {payload}")

        json_str = json.dumps(payload, ensure_ascii=False)
        print(f"\n[客户端] JSON 序列化:")
        print(f"  JSON 字符串: {repr(json_str)}")
        print(f"  JSON 长度: {len(json_str)}")
        print(f"  JSON 编码: {json_str.encode('utf-8')}")

        await self.websocket.send(json_str)
        print(f"\n[客户端] 消息已发送")

        response = await self.websocket.recv()
        print(f"\n[客户端] 收到响应:")
        print(f"  响应类型: {type(response)}")
        print(f"  响应长度: {len(response)}")
        print(f"  响应内容: {repr(response)}")

        try:
            response_data = json.loads(response)
            print(f"\n[客户端] JSON 解析成功:")
            print(f"  响应数据: {response_data}")
            return response_data
        except json.JSONDecodeError as e:
            print(f"\n[客户端] JSON 解析失败: {e}")
            return None

    async def test_normal_message(self):
        """测试普通中文消息"""
        payload = {
            "type": "chat",
            "message": "你好，请介绍一下自己",
            "model": "zyantine-v1"
        }
        await self.send_and_receive(payload, "测试 1: 普通中文消息")

    async def test_special_chars(self):
        """测试特殊字符"""
        payload = {
            "type": "chat",
            "message": "测试特殊字符：@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`",
            "model": "zyantine-v1"
        }
        await self.send_and_receive(payload, "测试 2: 特殊字符")

    async def test_emoji(self):
        """测试表情符号"""
        payload = {
            "type": "chat",
            "message": "你好！😊🎉🚀 今天天气怎么样？",
            "model": "zyantine-v1"
        }
        await self.send_and_receive(payload, "测试 3: 表情符号")

    async def test_long_message(self):
        """测试长消息"""
        payload = {
            "type": "chat",
            "message": "这是一个很长的消息，用于测试系统在处理长文本时的表现。" * 10,
            "model": "zyantine-v1"
        }
        await self.send_and_receive(payload, "测试 4: 长消息")

    async def test_mixed_language(self):
        """测试混合语言"""
        payload = {
            "type": "chat",
            "message": "Hello 你好，こんにちは 안녕하세요",
            "model": "zyantine-v1"
        }
        await self.send_and_receive(payload, "测试 5: 混合语言")

    async def test_empty_message(self):
        """测试空消息"""
        payload = {
            "type": "chat",
            "message": "",
            "model": "zyantine-v1"
        }
        await self.send_and_receive(payload, "测试 6: 空消息")

    async def test_invalid_json(self):
        """测试无效 JSON"""
        print(f"\n{'='*60}")
        print("测试 7: 无效 JSON")
        print(f"{'='*60}")

        invalid_json = "{invalid json"
        print(f"\n[客户端] 发送无效 JSON: {repr(invalid_json)}")

        await self.websocket.send(invalid_json)

        response = await self.websocket.recv()
        print(f"\n[客户端] 收到响应: {repr(response)}")

        try:
            response_data = json.loads(response)
            print(f"[客户端] 响应数据: {response_data}")
        except json.JSONDecodeError as e:
            print(f"[客户端] JSON 解析失败: {e}")

    async def test_unknown_type(self):
        """测试未知消息类型"""
        payload = {
            "type": "unknown_type",
            "data": "test"
        }
        await self.send_and_receive(payload, "测试 8: 未知消息类型")

    async def test_ping(self):
        """测试 ping"""
        payload = {"type": "ping"}
        await self.send_and_receive(payload, "测试 9: Ping")

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            print("\n连接已关闭")


async def run_all_tests(uri: str):
    """运行所有测试"""
    debugger = MessageDebugger(uri, client_id="debug_client")

    if not await debugger.connect():
        print("无法连接到服务器，测试终止")
        return

    try:
        await debugger.test_normal_message()
        await asyncio.sleep(1)

        await debugger.test_special_chars()
        await asyncio.sleep(1)

        await debugger.test_emoji()
        await asyncio.sleep(1)

        await debugger.test_long_message()
        await asyncio.sleep(1)

        await debugger.test_mixed_language()
        await asyncio.sleep(1)

        await debugger.test_empty_message()
        await asyncio.sleep(1)

        await debugger.test_invalid_json()
        await asyncio.sleep(1)

        await debugger.test_unknown_type()
        await asyncio.sleep(1)

        await debugger.test_ping()

        print(f"\n{'='*60}")
        print("所有测试完成")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await debugger.close()


async def interactive_debug(uri: str):
    """交互式调试模式"""
    debugger = MessageDebugger(uri, client_id="interactive_debug")

    if not await debugger.connect():
        print("无法连接到服务器")
        return

    print("\n进入交互式调试模式")
    print("输入消息内容，系统会显示详细的传输信息")
    print("输入 'quit' 退出\n")

    try:
        while True:
            user_input = input("\n请输入消息: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("退出调试模式")
                break

            if not user_input:
                continue

            payload = {
                "type": "chat",
                "message": user_input,
                "model": "zyantine-v1"
            }

            response = await debugger.send_and_receive(payload, f"用户消息: {user_input}")

            if response and response.get("type") == "response":
                print(f"\n[最终] AI 响应: {response.get('message')}")
            elif response and response.get("type") == "error":
                print(f"\n[最终] 错误: {response.get('message')}")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        await debugger.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket 消息调试工具")

    parser.add_argument("--uri", "-u", default="ws://localhost:8001/ws", help="WebSocket 服务器地址")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")
    parser.add_argument("--client-id", "-c", default="debug_client", help="客户端 ID")

    args = parser.parse_args()

    uri = f"{args.uri}?client_id={args.client_id}"

    if args.interactive:
        asyncio.run(interactive_debug(uri))
    else:
        asyncio.run(run_all_tests(uri))


if __name__ == "__main__":
    main()
