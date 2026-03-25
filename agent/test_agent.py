"""
测试 ECAPA-TDNN 超参数优化智能体系统

这个脚本用于测试智能体的各个功能模块
"""

import sys
import os
import dotenv
# load environment variables from .env file
dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("ZHIU_API_BASE_URL")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#忽略报错
import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")
warnings.filterwarnings("ignore", message="torchvision is not available - cannot save figures")

# 添加父目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpo_agent import (
    modify_config, 
    read_config, 
    backup_config,
    get_training_logs,
    run_training,
    run_evaluation,
    view_experiment_history,
    get_best_experiment,
    EXPERIMENTS_FILE,
    EXPERIMENTS_DIR
)

def test_read_config():
    """测试读取配置功能"""
    print("=" * 80)
    print("测试 1: 读取当前配置")
    print("=" * 80)
    result = read_config.invoke({})
    print(result)
    print()
    return True

def test_backup_config():
    """测试备份配置功能"""
    print("=" * 80)
    print("测试 2: 备份当前配置")
    print("=" * 80)
    result = backup_config.invoke({})
    print(result)
    print()
    return True

def test_modify_config():
    """测试修改配置功能"""
    print("=" * 80)
    print("测试 3: 修改配置（预览模式）")
    print("=" * 80)
    
    # 测试简单参数修改
    test_cases = [
        ('{"lr": 0.025}', False),  # 预览模式
        ('{"batch_size": 64}', False),
        ('{"number_of_epochs": 15}', False),
        ('{"compute_cost": {"loss_fn": {"margin": 0.3}}}', False),
    ]
    
    for config_json, persist in test_cases:
        print(f"修改配置: {config_json} (persist={persist})")
        result = modify_config.invoke({"config_json": config_json, "persist": persist})
        print(f"结果: {result}\n")
    
    print("=" * 80)
    print("测试 4: 实际修改配置（persist=True）")
    print("=" * 80)
    # 恢复原始值
    result = modify_config.invoke({"config_json": '{"lr": 0.001, "batch_size": 32}', "persist": True})
    print(result)
    print()
    return True

def test_get_training_logs():
    """测试获取训练日志功能"""
    print("=" * 80)
    print("测试 5: 获取训练日志")
    print("=" * 80)
    result = get_training_logs.invoke({})
    print(result)
    print()
    return True

def test_view_experiment_history():
    """测试查看实验历史功能"""
    print("=" * 80)
    print("测试 6: 查看实验历史")
    print("=" * 80)
    
    # 检查实验目录是否存在
    print(f"实验目录: {EXPERIMENTS_DIR}")
    print(f"实验文件: {EXPERIMENTS_FILE}")
    print()
    
    result = view_experiment_history.invoke({"n": 5})
    print(result)
    print()
    return True

def test_get_best_experiment():
    """测试获取最佳实验功能"""
    print("=" * 80)
    print("测试 7: 获取最佳实验（按EER）")
    print("=" * 80)
    
    result = get_best_experiment.invoke({"metric": "eer"})
    print(result)
    print()
    
    print("=" * 80)
    print("测试 8: 获取最佳实验（按准确率）")
    print("=" * 80)
    
    result = get_best_experiment.invoke({"metric": "accuracy"})
    print(result)
    print()
    return True

def test_run_training():
    """测试训练功能"""
    # print("=" * 80)
    # print("测试 9: 运行训练")
    # print("=" * 80)
    
    # print("⚠️  注意：此测试将实际运行训练，可能需要较长时间")
    # print("训练命令使用 subprocess.run() 执行：")
    # print('  subprocess.run(["python", TRAIN_SCRIPT, CONFIG_PATH], ...)')
    # print()
    
    # # 说明GPU指定方法
    # print("💡 指定GPU的方法:")
    # print("  方法1: 设置环境变量")
    # print('    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU 0')
    # print()
    # print("  方法2: 在命令中添加环境变量")
    # print('    subprocess.run(["CUDA_VISIBLE_DEVICES=0", "python", ...])')
    # print()
    
    # 询问用户是否继续
    user_input = input("是否继续运行训练？(y/n): ").strip().lower()
    
    if user_input == 'y':
        print("\n开始训练...")
        result = run_training.invoke({})
        print("\n训练结果:")
        print(result)
        print()
        return True
    else:
        print("⏭️  跳过训练测试")
        print()
        return True

def test_run_evaluation():
    """测试评估功能"""
    print("=" * 80)
    print("测试 10: 运行评估")
    print("=" * 80)
    
    print("⚠️  注意：此测试将运行模型评估")
    print("评估命令使用 subprocess.run() 执行")
    print()
    
    user_input = input("是否继续运行评估？(y/n): ").strip().lower()
    
    if user_input == 'y':
        print("\n开始评估...")
        result = run_evaluation.invoke({})
        print("\n评估结果:")
        print(result)
        print()
        return True
    else:
        print("⏭️  跳过评估测试")
        print()
        return True

def test_agent_initialization():
    """测试智能体初始化"""
    print("=" * 80)
    print("测试 9: 智能体初始化 (LangChain 1.0)")
    print("=" * 80)
    
    try:
        from hpo_agent import create_agent, load_system_prompt
        
        # 测试系统提示词加载
        print("加载系统提示词...")
        prompt = load_system_prompt()
        print(f"✅ 系统提示词加载成功，长度: {len(prompt)} 字符")
        
        # 测试智能体创建
        print("\n创建智能体 (LangChain 1.0 新方式)...")
        agent = create_agent()
        print("✅ 智能体创建成功")
        
        # 测试新的 invoke 格式
        print("\n测试新的 invoke 格式 (messages)...")
        test_message = {
            "messages": [
                {"role": "user", "content": "读取当前配置"}
            ]
        }
        print(f"输入: {test_message}")
        result = agent.invoke(test_message)
        print(f"✅ 调用成功，返回结果类型: {type(result)}; 内容预览: {result[-1:]}")
        return True
    except Exception as e:
        print(f"❌ 智能体初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n")
    print("🧪 ECAPA-TDNN 超参数优化智能体系统测试")
    print("=" * 80)
    print()
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("读取配置", test_read_config()))
    # test_results.append(("备份配置", test_backup_config()))
    # test_results.append(("修改配置", test_modify_config()))
    # test_results.append(("获取训练日志", test_get_training_logs()))
    # test_results.append(("查看实验历史", test_view_experiment_history()))
    # test_results.append(("获取最佳实验", test_get_best_experiment()))
    # test_results.append(("运行训练", test_run_training()))
    # test_results.append(("运行评估", test_run_evaluation()))
    # test_results.append(("智能体初始化", test_agent_initialization()))
    
    # 汇总测试结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"总计: {len(test_results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    print()
    
    if failed == 0:
        print("🎉 所有测试通过！智能体系统已准备就绪。")
        print("\n可以运行以下命令启动智能体:")
        print("  cd agent")
        print("  python hpo_agent.py")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")
    
    print()

if __name__ == "__main__":
    main()