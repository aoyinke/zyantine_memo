"""
测试 API 接口
"""
import requests
import json
from typing import Optional


class APITester:
    """API 测试器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def test_health(self):
        """测试健康检查"""
        print("\n" + "="*60)
        print("测试健康检查接口")
        print("="*60)

        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"错误: {e}")
            return False

    def test_list_models(self):
        """测试获取模型列表"""
        print("\n" + "="*60)
        print("测试获取模型列表")
        print("="*60)

        try:
            response = requests.get(f"{self.base_url}/v1/models")
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"错误: {e}")
            return False

    def test_chat_completion(self, message: str = "你好，请介绍一下自己"):
        """测试聊天完成接口"""
        print("\n" + "="*60)
        print("测试聊天完成接口")
        print("="*60)

        payload = {
            "model": "zyantine-v1",
            "messages": [
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }

        print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"错误: {e}")
            return False

    def test_chat_completion_enhanced(self, message: str = "你好，请介绍一下自己"):
        """测试增强版聊天完成接口"""
        print("\n" + "="*60)
        print("测试增强版聊天完成接口")
        print("="*60)

        payload = {
            "model": "zyantine-enhanced",
            "messages": [
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }

        print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"错误: {e}")
            return False

    def test_chat_completion_stream(self, message: str = "你好，请介绍一下自己"):
        """测试流式聊天完成接口"""
        print("\n" + "="*60)
        print("测试流式聊天完成接口")
        print("="*60)

        payload = {
            "model": "zyantine-v1",
            "messages": [
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions/stream",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            print(f"状态码: {response.status_code}")
            print("流式响应:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(line_str)
                    if line_str == "data: [DONE]":
                        break
            return response.status_code == 200
        except Exception as e:
            print(f"错误: {e}")
            return False

    def test_conversation(self):
        """测试多轮对话"""
        print("\n" + "="*60)
        print("测试多轮对话")
        print("="*60)

        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！很高兴见到你。"},
            {"role": "user", "content": "你叫什么名字？"}
        ]

        payload = {
            "model": "zyantine-v1",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"错误: {e}")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("开始运行所有测试")
        print("="*60)

        results = {
            "健康检查": self.test_health(),
            "获取模型列表": self.test_list_models(),
            "聊天完成": self.test_chat_completion(),
            "增强版聊天完成": self.test_chat_completion_enhanced(),
            "流式聊天完成": self.test_chat_completion_stream(),
            "多轮对话": self.test_conversation()
        }

        print("\n" + "="*60)
        print("测试结果汇总")
        print("="*60)
        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")

        passed = sum(1 for r in results.values() if r)
        total = len(results)
        print(f"\n总计: {passed}/{total} 测试通过")

        return all(results.values())


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试自衍体 AI API")
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="API 服务地址")
    parser.add_argument("--test", "-t", choices=["health", "models", "chat", "enhanced", "stream", "conversation", "all"],
                       default="all", help="测试类型")

    args = parser.parse_args()

    tester = APITester(base_url=args.url)

    if args.test == "all":
        success = tester.run_all_tests()
    elif args.test == "health":
        success = tester.test_health()
    elif args.test == "models":
        success = tester.test_list_models()
    elif args.test == "chat":
        success = tester.test_chat_completion()
    elif args.test == "enhanced":
        success = tester.test_chat_completion_enhanced()
    elif args.test == "stream":
        success = tester.test_chat_completion_stream()
    elif args.test == "conversation":
        success = tester.test_conversation()

    if success:
        print("\n✓ 所有测试通过")
        exit(0)
    else:
        print("\n✗ 部分测试失败")
        exit(1)


if __name__ == "__main__":
    main()
