"""
WebSocket 测试脚本
用于测试 API 服务器的 WebSocket 功能
"""
import asyncio
import json
import argparse
import websockets
from typing import Optional


class WebSocketTestClient:
    """WebSocket 测试客户端"""

    def __init__(self, uri: str, client_id: str = "test_client"):
        self.uri = uri
        self.client_id = client_id
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """连接到服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            print(f"已连接到服务器: {self.uri}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    async def send_chat_message(self, message: str, model: str = "zyantine-v1"):
        """发送聊天消息"""
        if not self.websocket:
            print("未连接到服务器")
            return None

        payload = {
            "type": "chat",
            "message": message,
            "model": model
        }

        await self.websocket.send(json.dumps(payload, ensure_ascii=False))
        print(f"发送消息: {message}")

        response = await self.websocket.recv()
        response_data = json.loads(response)
        return response_data

    async def send_ping(self):
        """发送 ping 消息"""
        if not self.websocket:
            print("未连接到服务器")
            return None

        payload = {"type": "ping"}
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))
        print("发送 ping")

        response = await self.websocket.recv()
        response_data = json.loads(response)
        return response_data

    async def send_close(self):
        """发送关闭消息"""
        if not self.websocket:
            print("未连接到服务器")
            return None

        payload = {"type": "close"}
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))
        print("发送关闭请求")

        response = await self.websocket.recv()
        response_data = json.loads(response)
        return response_data

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            print("连接已关闭")


async def test_basic_chat(client: WebSocketTestClient):
    """测试基本聊天功能"""
    print("\n" + "="*60)
    print("测试基本聊天功能")
    print("="*60)

    messages = [
        "你好，请介绍一下自己",
        "你有什么功能？",
        "今天天气怎么样？"
    ]

    for msg in messages:
        print(f"\n用户: {msg}")
        response = await client.send_chat_message(msg)
        
        if response and response.get("type") == "response":
            print(f"AI: {response.get('message')}")
            print(f"模型: {response.get('model')}")
            print(f"时间戳: {response.get('timestamp')}")
        elif response and response.get("type") == "error":
            print(f"错误: {response.get('message')}")
        
        await asyncio.sleep(1)


async def test_ping_pong(client: WebSocketTestClient):
    """测试 ping/pong 功能"""
    print("\n" + "="*60)
    print("测试 ping/pong 功能")
    print("="*60)

    response = await client.send_ping()
    
    if response and response.get("type") == "pong":
        print(f"收到 pong 响应，时间戳: {response.get('timestamp')}")
    else:
        print(f"意外响应: {response}")


async def test_enhanced_model(client: WebSocketTestClient):
    """测试增强版模型"""
    print("\n" + "="*60)
    print("测试增强版模型")
    print("="*60)

    message = "请用更详细的方式回答这个问题：什么是人工智能？"
    print(f"用户: {message}")
    
    response = await client.send_chat_message(message, model="zyantine-enhanced")
    
    if response and response.get("type") == "response":
        print(f"AI (增强版): {response.get('message')}")
    elif response and response.get("type") == "error":
        print(f"错误: {response.get('message')}")


async def test_error_handling(client: WebSocketTestClient):
    """测试错误处理"""
    print("\n" + "="*60)
    print("测试错误处理")
    print("="*60)

    invalid_payload = {"type": "invalid_type"}
    await client.websocket.send(json.dumps(invalid_payload, ensure_ascii=False))
    print("发送无效消息类型")

    response = await client.websocket.recv()
    response_data = json.loads(response)
    
    if response_data.get("type") == "error":
        print(f"正确捕获错误: {response_data.get('message')}")
    else:
        print(f"意外响应: {response_data}")


async def test_close_connection(client: WebSocketTestClient):
    """测试关闭连接"""
    print("\n" + "="*60)
    print("测试关闭连接")
    print("="*60)

    await client.send_close()
    await asyncio.sleep(0.5)


async def run_all_tests(uri: str):
    """运行所有测试"""
    client = WebSocketTestClient(uri, client_id="test_client")

    if not await client.connect():
        print("无法连接到服务器，测试终止")
        return

    try:
        await test_ping_pong(client)
        await test_basic_chat(client)
        await test_enhanced_model(client)
        await test_error_handling(client)
        await test_close_connection(client)

        print("\n" + "="*60)
        print("所有测试完成")
        print("="*60)

    except Exception as e:
        print(f"测试过程中出错: {e}")
    finally:
        await client.close()


async def interactive_mode(uri: str):
    """交互模式"""
    client = WebSocketTestClient(uri, client_id="interactive_client")

    if not await client.connect():
        print("无法连接到服务器")
        return

    print("\n进入交互模式（输入 'quit' 退出）")
    print("="*60)

    try:
        while True:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("退出交互模式")
                break

            if not user_input:
                continue

            response = await client.send_chat_message(user_input)
            
            if response and response.get("type") == "response":
                print(f"AI: {response.get('message')}")
            elif response and response.get("type") == "error":
                print(f"错误: {response.get('message')}")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        await client.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WebSocket 测试客户端")

    parser.add_argument("--uri", "-u", default="ws://localhost:8001/ws", help="WebSocket 服务器地址")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")
    parser.add_argument("--client-id", "-c", default="test_client", help="客户端 ID")

    args = parser.parse_args()

    uri = f"{args.uri}?client_id={args.client_id}"

    if args.interactive:
        asyncio.run(interactive_mode(uri))
    else:
        asyncio.run(run_all_tests(uri))


if __name__ == "__main__":
    main()
